import os
import logging
import datetime
import threading
import sys

# Set up logging to a file in the same directory as this script
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'markov_debug.log')
logging.basicConfig(
    level=logging.DEBUG,
    filename=log_path,
    filemode='w',  # Overwrite each run; use 'a' to append
    format='%(asctime)s %(levelname)s: %(message)s'
)

from typing import List, Tuple

from TwitchWebsocket import Message, TwitchWebsocket
from nltk.tokenize import sent_tokenize
import socket, time, re, string
import random

from Settings import Settings, SettingsData
from Database import Database
from Timer import LoopingTimer
from Tokenizer import detokenize, tokenize

from Log import Log
Log(__file__)

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def contains_private_unicode(key):
    if not key:
        return False
    for word in key:
        if any(0xE0000 <= ord(char) <= 0xE007F for char in word):
            return True
    return False

class MarkovChain:
    def __init__(self):
        self.prev_message_t = 0
        self._enabled = True
        # This regex should detect similar phrases as links as Twitch does
        self.link_regex = re.compile("\w+\.[a-z]{2,}")
        # List of moderators used in blacklist modification, includes broadcaster
        self.mod_list = []
        self.set_blacklist()
        self.last_trigger_time = 0
        self._randomized_timer_thread = None
        self._randomized_timer_stop = threading.Event()

        # Fill previously initialised variables with data from the settings.txt file
        Settings(self)
        self.db = Database(self.chan)

        # Set up daemon Timer to send help messages
        if self.help_message_timer > 0:
            if self.help_message_timer < 300:
                raise ValueError("Value for \"HelpMessageTimer\" in must be at least 300 seconds, or a negative number for no help messages.")
            t = LoopingTimer(self.help_message_timer, self.send_help_message)
            t.start()
        
        # Set up daemon Timer to send automatic or randomized generation messages
        if getattr(self, 'randomized_generation_timer_enabled', False):
            self._start_randomized_generation_timer()
        elif self.automatic_generation_timer > 0:
            if self.automatic_generation_timer < 5:
                raise ValueError("Value for \"AutomaticGenerationMessage\" in must be at least 5 seconds, or a negative number for no automatic generations.")
            t = LoopingTimer(self.automatic_generation_timer, self.send_automatic_generation_message)
            t.start()

        # Set up daemon Timer to check for manual triggers from GUI
        t = LoopingTimer(1, lambda: self.check_for_manual_trigger())
        t.start()
        # Start TCP trigger server in a background thread
        threading.Thread(target=self.tcp_trigger_server, daemon=True).start()

        self.ws = TwitchWebsocket(host=self.host, 
                                  port=self.port,
                                  chan=self.chan,
                                  nick=self.nick,
                                  auth=self.auth,
                                  callback=self.message_handler,
                                  capability=["commands", "tags"],
                                  live=True)
        timestamp = datetime.datetime.now().strftime("[%m/%d/%Y - %H:%M:%S]:")
        print(f"{timestamp} [TwitchWebsocket.TwitchWebsocket] [INFO    ] - Attempting to initialize websocket connection.", flush=True)
        self.ws.start_bot()

    def set_settings(self, settings: SettingsData):
        """Fill class instance attributes based on the settings file."""
        self.host = settings["Host"]
        self.port = settings["Port"]
        self.chan = settings["Channel"]
        self.nick = settings["Nickname"]
        self.auth = settings["Authentication"]
        self.denied_users = [user.lower() for user in settings["DeniedUsers"]] + [self.nick.lower()]
        self.allowed_users = [user.lower() for user in settings["AllowedUsers"]]
        self.cooldown = int(settings["Cooldown"])
        self.key_length = settings["KeyLength"]
        self.max_sentence_length = settings["MaxSentenceWordAmount"]
        self.min_sentence_length = settings["MinSentenceWordAmount"]
        self.help_message_timer = settings["HelpMessageTimer"]
        self.automatic_generation_timer = settings["AutomaticGenerationTimer"]
        self.whisper_cooldown = settings["WhisperCooldown"]
        self.enable_generate_command = settings["EnableGenerateCommand"]
        self.sent_separator = settings["SentenceSeparator"]
        self.allow_generate_params = settings["AllowGenerateParams"]
        self.generate_commands = tuple(settings["GenerateCommands"])
        self.cooldown_warned = False
        # New randomized timer fields
        self.randomized_generation_timer_enabled = settings.get("RandomizedGenerationTimerEnabled", False)
        self.randomized_generation_timer_min = settings.get("RandomizedGenerationTimerMin", 30)
        self.randomized_generation_timer_max = settings.get("RandomizedGenerationTimerMax", 100)

    def _start_randomized_generation_timer(self):
        if self._randomized_timer_thread and self._randomized_timer_thread.is_alive():
            return  # Already running
        self._randomized_timer_stop.clear()
        self._randomized_timer_thread = threading.Thread(target=self._randomized_timer_loop, daemon=True)
        self._randomized_timer_thread.start()

    def _randomized_timer_loop(self):
        while not self._randomized_timer_stop.is_set() and getattr(self, 'randomized_generation_timer_enabled', False):
            min_val = max(5, self.randomized_generation_timer_min)
            wait_time = random.randint(min_val, self.randomized_generation_timer_max)
            logger.info(f"[RandomizedTimer] Next message in {wait_time} seconds.")
            timestamp = datetime.datetime.now().strftime("[%m/%d/%Y - %H:%M:%S]")
            print(f"{timestamp}: (RandomizedTimer) Selected wait time: {wait_time} seconds.", flush=True)
            for _ in range(wait_time):
                if self._randomized_timer_stop.is_set() or not getattr(self, 'randomized_generation_timer_enabled', False):
                    return
                time.sleep(1)
            self.send_randomized_generation_message()

    def message_handler(self, m: Message, check_trigger=True):
        try:
            if check_trigger:
                self.check_for_manual_trigger()
            
            # Check if the message type indicates a successful channel join
            if m.type == "366":
                timestamp = datetime.datetime.now().strftime("[%m/%d/%Y - %H:%M:%S]:")
                logger.info(f"Successfully joined channel: #{m.channel}")
                print(f"{timestamp} Successfully joined channel: #{m.channel}", flush=True)
                # Request the list of moderators for modifying the blacklist
                logger.info("Fetching mod list...")
                print(f"{timestamp} Fetching mod list...", flush=True)
                self.ws.send_message("/mods")

            # Handle NOTICE messages, which may contain moderator information
            elif m.type == "NOTICE":
                if m.message.startswith("The moderators of this channel are:"):
                    # Extract the list of moderators from the message
                    string_list = m.message.replace("The moderators of this channel are:", "").strip()
                    self.mod_list = [m.channel] + string_list.split(", ")
                    logger.info(f"Fetched mod list. Found {len(self.mod_list) - 1} mods.")
                elif m.message == "There are no moderators of this channel.":
                    self.mod_list = [m.channel]
                    logger.info(f"Fetched mod list. Found no mods.")
                else:
                    logger.info(m.message)

            # Handle user messages (PRIVMSG and WHISPER)
            elif m.type in ("PRIVMSG", "WHISPER"):
                # Check if the message is a command to enable the generate feature
                if m.message.startswith("!enable") and self.check_if_permissions(m):
                    if self._enabled:
                        self.ws.send_whisper(m.user, "The generate command is already enabled.")
                    else:
                        self.ws.send_whisper(m.user, "Users can now use generate command again.")
                        self._enabled = True
                        logger.info("Users can now use generate command again.")

                # Check if the message is a command to disable the generate feature
                elif m.message.startswith("!disable") and self.check_if_permissions(m):
                    if self._enabled:
                        self.ws.send_whisper(m.user, "Users can now no longer use generate command.")
                        self._enabled = False
                        logger.info("Users can now no longer use generate command.")
                    else:
                        self.ws.send_whisper(m.user, "The generate command is already disabled.")

                # Check if the message is a command to set the cooldown
                elif m.message.startswith(("!setcooldown", "!setcd")) and self.check_if_permissions(m):
                    split_message = m.message.split(" ")
                    if len(split_message) == 2:
                        try:
                            cooldown = int(split_message[1])  # Convert the cooldown value to an integer
                        except ValueError:
                            self.ws.send_whisper(m.user, f"The parameter must be an integer amount, eg: !setcd 30")
                            return
                    self.cooldown = cooldown
                    Settings.update_cooldown(cooldown)  # Update the cooldown in settings
                    self.ws.send_whisper(m.user, f"The !generate cooldown has been set to {cooldown} seconds.")
                else:
                    self.ws.send_whisper(m.user, f"Please add exactly 1 integer parameter, eg: !setcd 30.")

                # Process regular chat messages
                if m.type == "PRIVMSG":
                    # Ignore messages from users in the denied users list
                    if m.user.lower() in self.denied_users:
                        return
                    
                    # Check if the message is a generate command
                    if self.check_if_generate(m.message):
                        # If generate command is disabled and user lacks permissions, ignore the message
                        if not self.enable_generate_command and not self.check_if_permissions(m):
                            return

                        # If the generate feature is disabled, send a whisper to the user
                        if not self._enabled:
                            if not self.db.check_whisper_ignore(m.user):
                                self.send_whisper(m.user, "The !generate has been turned off. !nopm to stop me from whispering you.")
                            return

                        cur_time = time.time()
                        remaining = int(self.prev_message_t + self.cooldown - cur_time)
                        if self.prev_message_t + self.cooldown < cur_time or self.check_if_permissions(m):
                            # Check if the message passes the filter
                            if self.check_filter(m.message):
                                sentence = "You can't make me say that, you madman!"
                            else:
                                # Tokenize the message to extract parameters for generation
                                params = tokenize(m.message)[2:] if self.allow_generate_params else None
                                # Generate a sentence based on the parameters
                                sentence, success = self.generate(params)
                                if success:
                                    # Reset the previous message time if a message was generated
                                    self.prev_message_t = time.time()
                                    self.cooldown_warned = False  # Reset warning flag after successful generation
                            logger.info(sentence)  # Log the generated sentence
                            self.ws.send_message(sentence)  # Send the generated sentence to the chat
                            timestamp = datetime.datetime.now().strftime("[%m/%d/%Y - %H:%M:%S]:")
                            print(f"{timestamp} {sentence}", flush=True)  # Print with timecode for GUI
                            logging.debug(f"[DEBUG] About to send message to Twitch: {repr(sentence)}")
                            logging.debug(f"[DEBUG] Message sent successfully")
                        else:
                            if not self.cooldown_warned:
                                self.ws.send_message(f"Yap Bot's on cooldown for {max(1, remaining)} seconds")
                                self.cooldown_warned = True
                            # If cooldown is active, inform the user
                            if not self.db.check_whisper_ignore(m.user):
                                self.send_whisper(m.user, f"Cooldown hit: {self.prev_message_t + self.cooldown - cur_time:0.2f} out of {self.cooldown:.0f}s remaining. !nopm to stop these cooldown pm's.")
                            logger.info(f"Cooldown hit with {self.prev_message_t + self.cooldown - cur_time:0.2f}s remaining.")
                            timestamp = datetime.datetime.now().strftime("[%m/%d/%Y - %H:%M:%S]:")
                            print(f"{timestamp} Cooldown hit with {self.prev_message_t + self.cooldown - cur_time:0.2f}s remaining.", flush=True)
                        return
                    
                    # Handle help command
                    elif m.message.startswith(("!ghelp", "!genhelp", "!generatehelp")):
                        self.send_help_message()

                    # Ignore the message if it is deemed a command
                    elif self.check_if_other_command(m.message):
                        return
                    
                    # Ignore the message if it contains a link
                    elif self.check_link(m.message):
                        return

                    # Process emotes in the message
                    if "emotes" in m.tags:
                        # Ignore messages containing bit emotes
                        if "emotesv2_" in m.tags["emotes"]:
                            return

                        # Replace modified emotes with normal versions
                        for modifier in self.extract_modifiers(m.tags["emotes"]):
                            m.message = m.message.replace(modifier, "")

                    # Check if the message contains any blacklisted words
                    if self.check_filter(m.message):
                        logger.warning(f"Sentence contained blacklisted word or phrase:\"{m.message}\"")
                        return
                    
                    else:
                        # Split the message into sentences using NLTK's tokenizer
                        try:
                            sentences = sent_tokenize(m.message.strip(), language="english")
                            logger.info(f"Successfully split message into sentences: {sentences}")
                        except LookupError:
                            logger.debug("Downloading required punkt resource...")
                            import nltk
                            nltk.download('punkt')  # Download the punkt tokenizer resource
                            logger.debug("Downloaded required punkt resource.")
                            sentences = sent_tokenize(m.message.strip(), language="english")
                            logger.info(f"Successfully split message into sentences after downloading resource: {sentences}")

                        # Process each sentence
                        for sentence in sentences:
                            logger.info(f"Processing sentence: {sentence}")  # Log the current sentence being processed
                            
                            # Tokenize the sentence into words
                            words = tokenize(sentence)
                            logger.info(f"Tokenized sentence into words: {words}")  # Log the tokenized words
                            
                            # Remove empty words caused by double spaces
                            if "" in words:
                                words = [word for word in words if word]
                                logger.info(f"Removed empty words, resulting in: {words}")  # Log the cleaned words

                            # Ignore sentences that are too short
                            if len(words) <= self.key_length:
                                logger.info(f"Ignored sentence due to short length: {sentence}")
                                continue
                            
                            # Add a new starting point for a sentence to the database
                            self.db.add_start_queue([words[x] for x in range(self.key_length)])
                            logger.info(f"Added starting point to database: {[words[x] for x in range(self.key_length)]}")

                            # Create a key variable for the grammar dictionary
                            key = list()
                            for word in words:
                                # Set up key for first use
                                if len(key) < self.key_length:
                                    key.append(word)
                                    continue
                                
                                # Add the current key and word to the database
                                self.db.add_rule_queue(key + [word])
                                logger.info(f"Added rule to database: {key + [word]}")  # Log the rule added
                                
                                # Update the key for the next word
                                key.pop(0)
                                key.append(word)
                            
                            # Add <END> at the end of the sentence
                            self.db.add_rule_queue(key + ["<END>"])
                            logger.info(f"Added end marker to database for sentence: {key + ['<END>']}")
                        
                elif m.type == "WHISPER":
                    # Allow users to manage their whisper settings
                    if m.message == "!nopm":
                        logger.debug(f"Adding {m.user} to Do Not Whisper.")
                        self.db.add_whisper_ignore(m.user)
                        self.ws.send_whisper(m.user, "You will no longer be sent whispers. Type !yespm to reenable. ")

                    elif m.message == "!yespm":
                        logger.debug(f"Removing {m.user} from Do Not Whisper.")
                        self.db.remove_whisper_ignore(m.user)
                        self.ws.send_whisper(m.user, "You will again be sent whispers. Type !nopm to disable again. ")

                    # Allow moderators and allowed users to manage the blacklist
                    elif m.user.lower() in self.mod_list + ["cubiedev"] + self.allowed_users:
                        # Adding to the blacklist
                        if self.check_if_our_command(m.message, "!blacklist"):
                            if len(m.message.split()) == 2:
                                word = m.message.split()[1].lower()
                                self.blacklist.append(word)
                                logger.info(f"Added `{word}` to Blacklist.")
                                self.write_blacklist(self.blacklist)
                                self.ws.send_whisper(m.user, "Added word to Blacklist.")
                            else:
                                self.ws.send_whisper(m.user, "Expected Format: `!blacklist word` to add `word` to the blacklist")

                        # Removing from the blacklist
                        elif self.check_if_our_command(m.message, "!whitelist"):
                            if len(m.message.split()) == 2:
                                word = m.message.split()[1].lower()
                                try:
                                    self.blacklist.remove(word)
                                    logger.info(f"Removed `{word}` from Blacklist.")
                                    self.write_blacklist(self.blacklist)
                                    self.ws.send_whisper(m.user, "Removed word from Blacklist.")
                                except ValueError:
                                    self.ws.send_whisper(m.user, "Word was already not in the blacklist.")
                            else:
                                self.ws.send_whisper(m.user, "Expected Format: `!whitelist word` to remove `word` from the blacklist.")
                        
                        # Checking whether a word is in the blacklist
                        elif self.check_if_our_command(m.message, "!check"):
                            if len(m.message.split()) == 2:
                                word = m.message.split()[1].lower()
                                if word in self.blacklist:
                                    self.ws.send_whisper(m.user, "This word is in the Blacklist.")
                                else:
                                    self.ws.send_whisper(m.user, "This word is not in the Blacklist.")
                            else:
                                self.ws.send_whisper(m.user, "Expected Format: `!check word` to check whether `word` is on the blacklist.")

                elif m.type == "CLEARMSG":
                    # If a message is deleted, its contents will be unlearned
                    self.db.unlearn(m.message)

        except Exception as e:
            logger.exception(e)
            timestamp = datetime.datetime.now().strftime("[%m/%d/%Y - %H:%M:%S]:")
            print(f"{timestamp} Exception: {e}", flush=True)

    def generate(self, params: List[str] = None) -> "Tuple[str, bool]":
        """Given an input sentence, generate the remainder of the sentence using the learned data.

        Args:
            params (List[str]): A list of words to use as an input to use as the start of generating.
        
        Returns:
            Tuple[str, bool]: A tuple of a sentence as the first value, and a boolean indicating
                whether the generation succeeded as the second value.
        """
        if params is None:
            params = []

        logging.debug(f"[DEBUG] generate called with params: {repr(params)}")

        # If any input param contains private use unicode, ignore input and run as default
        if any(contains_private_unicode([param]) for param in params):
            logging.debug(f"[DEBUG] User input contains private unicode, running default generate. Params: {repr(params)}")
            params = []

        # List of sentences that will be generated. In some cases, multiple sentences will be generated,
        # e.g. when the first sentence has less words than self.min_sentence_length.
        sentences = [[]]

        # Check for commands or recursion, eg: !generate !generate
        if len(params) > 0:
            if self.check_if_other_command(params[0]):
                return "You can't make me do commands, you madman!", False

        # Get the starting key and starting sentence.
        # If there is more than 1 param, get the last 2 as the key.
        # Note that self.key_length is fixed to 2 in this implementation
        if len(params) > 1:
            key = params[-self.key_length:]
            logging.debug(f"[DEBUG] Multi-word params, key: {repr(key)}")
            if contains_private_unicode(key):
                logging.debug(f"[DEBUG] Fallback: multi-word params contain private unicode: {repr(key)} | params: {repr(params)}")
                return f'I haven\'t extracted "{detokenize(key)}" from chat yet.', False
            sentences[0] = params.copy()
            logging.debug(f"[DEBUG] Multi-word params, using key: {repr(key)}")
        elif len(params) == 1:
            key = self.db.get_next_single_start(params[0])
            logging.debug(f"[DEBUG] get_next_single_start({params[0]}) returned: {repr(key)}")
            if key is not None and contains_private_unicode(key):
                logging.debug(f"[DEBUG] Fallback: get_next_single_start returned private unicode: {repr(key)} | params: {repr(params)}")
                key = None
            if key is None:
                key = self.db.get_next_single_initial(0, params[0])
                logging.debug(f"[DEBUG] get_next_single_initial(0, {params[0]}) returned: {repr(key)}")
                if key is not None and contains_private_unicode(key):
                    logging.debug(f"[DEBUG] Fallback: get_next_single_initial returned private unicode: {repr(key)} | params: {repr(params)}")
                    key = None
                if key is None:
                    logging.debug(f"[DEBUG] Fallback: haven't extracted {params[0]} from chat yet. | params: {repr(params)}")
                    return f'I haven\'t extracted "{params[0]}" from chat yet.', False
            sentences[0] = key.copy()
            logging.debug(f"[DEBUG] Single-word param, using key: {repr(key)}")
        else:
            all_keys = [k for k in self.db.get_all_starts() if k and not contains_private_unicode(k)]
            logging.debug(f"[DEBUG] All valid starting keys: {[repr(k) for k in all_keys]}")
            if all_keys:
                key = random.choice(all_keys)
                logging.debug(f"[DEBUG] Chosen starting key: {repr(key)}")
                sentences[0] = key.copy()
            else:
                logging.debug(f"[DEBUG] Fallback: not enough learned information. | params: {repr(params)}")
                return "There is not enough learned information yet.", False
        
        # Counter to prevent infinite loops (i.e. constantly generating <END> while below the 
        # minimum number of words to generate)
        i = 0
        while self.sentence_length(sentences) < self.max_sentence_length and i < self.max_sentence_length * 2:
            # Use key to get next word
            if i == 0:
                # Prevent fetching <END> on the first word
                word = self.db.get_next_initial(i, key)
            else:
                word = self.db.get_next(i, key)

            i += 1

            if word == "<END>" or word == None:
                # Break, unless we are before the min_sentence_length
                if i < self.min_sentence_length:
                    key = self.db.get_start()
                    # Ensure that the key can be generated. Otherwise we still stop.
                    if key:
                        # Start a new sentence
                        sentences.append([])
                        for entry in key:
                            sentences[-1].append(entry)
                        continue
                break

            # Otherwise add the word
            sentences[-1].append(word)
            
            # Shift the key so on the next iteration it gets the next item
            key.pop(0)
            key.append(word)
        
        # If there were params, but the sentence resulting is identical to the params
        # Then the params did not result in an actual sentence
        # If so, restart without params
        if len(params) > 0 and params == sentences[0]:
            logging.debug(f"[DEBUG] Fallback: params identical to generated sentence. Params: {repr(params)}, Sentence: {repr(sentences[0])}")
            return "I haven't learned what to do with \"" + detokenize(params[-self.key_length:]) + "\" yet.", False

        return self.sent_separator.join(detokenize(sentence) for sentence in sentences), True

    def sentence_length(self, sentences: List[List[str]]) -> int:
        """Given a list of tokens representing a sentence, return the number of words in there.

        Args:
            sentences (List[List[str]]): List of lists of tokens that make up a sentence,
                where a token is a word or punctuation. For example:
                [['Hello', ',', 'you', "'re", 'Tom', '!'], ['Yes', ',', 'I', 'am', '.']]
                This would return 6.

        Returns:
            int: The number of words in the sentence.
        """
        count = 0
        for sentence in sentences:
            for token in sentence:
                if token not in string.punctuation and token[0] != "'":
                    count += 1
        return count

    def extract_modifiers(self, emotes: str) -> List[str]:
        """Extract emote modifiers from emotes, such as the the horizontal flip.

        Args:
            emotes (str): String containing all emotes used in the message.
        
        Returns:
            List[str]: List of strings that show modifiers, such as "_HZ" for horizontal flip.
        """
        output = []
        try:
            while emotes:
                u_index = emotes.index("_")
                c_index = emotes.index(":", u_index)
                output.append(emotes[u_index:c_index])
                emotes = emotes[c_index:]
        except ValueError:
            pass
        return output

    def write_blacklist(self, blacklist: List[str]) -> None:
        """Write blacklist.txt given a list of banned words.

        Args:
            blacklist (List[str]): The list of banned words to write.
        """
        logger.debug("Writing Blacklist...")
        with open("blacklist.txt", "w") as f:
            f.write("\n".join(sorted(blacklist, key=lambda x: len(x), reverse=True)))
        logger.debug("Written Blacklist.")

    def set_blacklist(self) -> None:
        """Read blacklist.txt and set `self.blacklist` to the list of banned words."""
        logger.debug("Loading Blacklist...")
        try:
            with open("blacklist.txt", "r") as f:
                self.blacklist = [l.replace("\n", "") for l in f.readlines()]
                logger.debug("Loaded Blacklist.")
        
        except FileNotFoundError:
            logger.warning("Loading Blacklist Failed!")
            self.blacklist = ["<start>", "<end>"]
            self.write_blacklist(self.blacklist)

    def send_help_message(self) -> None:
        """Send a Help message to the connected chat, as long as the bot wasn't disabled."""
        if self._enabled:
            logger.info("Help message sent.")
            try:
                self.ws.send_message("Learn how this bot generates sentences here: https://github.com/CubieDev/TwitchMarkovChain#how-it-works")
            except socket.OSError as error:
                logger.warning(f"[OSError: {error}] upon sending help message. Ignoring.")

    def send_automatic_generation_message(self) -> None:
        """Send an automatic generation message to the connected chat.
        
        As long as the bot wasn't disabled, just like if someone typed "!g" in chat.
        """
        if self._enabled:
            sentence, success = self.generate()
            if success:
                logger.info(sentence)
                timestamp = datetime.datetime.now().strftime("[%m/%d/%Y - %H:%M:%S]:")
                print(f"{timestamp} Message generated by automatic timer:", flush=True)
                print(f"{timestamp} {sentence}", flush=True)
                # Try to send a message. Just log a warning on fail
                try:
                    self.ws.send_message(sentence)
                except socket.OSError as error:
                    logger.warning(f"[OSError: {error}] upon sending automatic generation message. Ignoring.")
            else:
                logger.info("Attempted to output automatic generation message, but there is not enough learned information yet.")

    def send_randomized_generation_message(self):
        """Send a randomized generation message to the connected chat, then schedule the next one."""
        if self._enabled:
            timestamp = datetime.datetime.now().strftime("[%m/%d/%Y - %H:%M:%S]")
            print(f"{timestamp}: (RandomizedTimer): Generating and sending message via randomized timer.", flush=True)
            sentence, success = self.generate()
            if success:
                print(f"{timestamp}: {sentence}", flush=True)
                logger.info(sentence)
                try:
                    self.ws.send_message(sentence)
                except socket.OSError as error:
                    logger.warning(f"[OSError: {error}] upon sending randomized generation message. Ignoring.")
            else:
                logger.info("Attempted to output randomized generation message, but there is not enough learned information yet.")

    def send_whisper(self, user: str, message: str) -> None:
        """Optionally send a whisper, only if "WhisperCooldown" is True.
        
        Args:
            user (str): The user to potentially whisper.
            message (str): The message to potentially whisper
        """
        if self.whisper_cooldown:
            self.ws.send_whisper(user, message)

    def check_filter(self, message: str) -> bool:
        """Returns True if message contains a banned word.
        
        Args:
            message (str): The message to check.
        """
        for word in tokenize(message):
            if word.lower() in self.blacklist:
                return True
        return False

    def check_if_our_command(self, message: str, *commands: "Tuple[str]") -> bool:
        """True if the first "word" of the message is in the tuple of commands

        Args:
            message (str): The message to check for a command.
            commands (Tuple[str]): A tuple of commands.

        Returns:
            bool: True if the first word in message is one of the commands.
        """
        return message.split()[0] in commands

    def check_if_generate(self, message: str) -> bool:
        """True if the first "word" of the message is one of the defined generate commands.

        Args:
            message (str): The message to check for the generate command (i.e !generate or !g).
        
        Returns:
            bool: True if the first word in message is a generate command.
        """
        return self.check_if_our_command(message, *self.generate_commands)
    
    def check_if_other_command(self, message: str) -> bool:
        """True if the message is any command, except /me. 

        Is used to avoid learning and generating commands.

        Args:
            message (str): The message to check.

        Returns:
            bool: True if the message is any potential command (starts with a '!', '/' or '.')
                with the exception of /me.
        """
        return message.startswith(("!", "/", ".")) and not message.startswith("/me")
    
    def check_if_permissions(self, m: Message) -> bool:
        """True if the user has heightened permissions.
        
        E.g. permissions to bypass cooldowns, update settings, disable the bot, etc.
        True for the streamer themselves, and the users set as the allowed users.

        Args:
            m (Message): The Message object that was sent from Twitch. 
                Has `user` and `channel` attributes.
        """
        return m.user == m.channel or m.user in self.allowed_users

    def check_link(self, message: str) -> bool:
        """True if `message` contains a link.

        Args:
            message (str): The message to check for a link.

        Returns:
            bool: True if the message contains a link.
        """
        return self.link_regex.search(message)

    def trigger_yap(self, user="ManualTrigger"):
        from datetime import datetime
        timestamp = datetime.now().strftime("[%m/%d/%Y - %H:%M:%S]:")
        print(f"{timestamp} Generate command triggered manually", flush=True)
        # Use the first generate command from settings
        command = self.generate_commands[0] if self.generate_commands else "!generate"
        class MockMessage:
            def __init__(self, user, message, channel, msg_type="PRIVMSG"):
                self.user = user
                self.message = message
                self.type = msg_type
                self.channel = channel  # Use the bot's channel
        # Create a mock message for the trigger
        mock_msg = MockMessage(user, command, self.chan)
        # Process it through the message handler
        self.message_handler(mock_msg, check_trigger=False)

    def check_for_manual_trigger(self):
        import time
        import glob
        from datetime import datetime
        trigger_files = glob.glob("trigger_yap_*.txt")
        for trigger_file in trigger_files:
            now = time.time()
            if now - self.last_trigger_time > 1.5:
                try:
                    os.remove(trigger_file)
                    timestamp = datetime.now().strftime("[%m/%d/%Y - %H:%M:%S]:")
                    print(f"{timestamp} Generate command trigerred manually", flush=True)
                    self.trigger_yap()
                    self.last_trigger_time = now
                except FileNotFoundError:
                    pass

    def tcp_trigger_server(self):
        import socket
        from datetime import datetime
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', 8765))
        server.listen(5)
        timestamp = datetime.now().strftime("[%m/%d/%Y - %H:%M:%S]:")
        print(f"{timestamp} TCP server for manual triggers listening on 127.0.0.1:8765", flush=True)
        while True:
            conn, _ = server.accept()
            try:
                data = conn.recv(1024)
                if data and b"YAP" in data:
                    self.trigger_yap()
            finally:
                conn.close()

if __name__ == "__main__":
    MarkovChain()