import datetime
from enum import Enum


class LogLevel(Enum):
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4
    VERBOSE = 5


class POILog:
    MAX_LOG_LEVEL = LogLevel.VERBOSE
    TAG_COLUMN_WIDTH = 18
    LEVEL_COLUMN_WIDTH = 18

    color_codes = {
        "bw": "\033[30;47m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "reset": "\033[0m",
    }

    color_map = {
        LogLevel.ERROR: color_codes["red"],
        LogLevel.WARN: color_codes["yellow"],
        LogLevel.INFO: color_codes["green"],
        LogLevel.DEBUG: color_codes["blue"],
        LogLevel.VERBOSE: color_codes["magenta"],
    }

    @staticmethod
    def apply_style(text, style_code):
        return f"{style_code}{text}{POILog.color_codes['reset']}"

    @staticmethod
    def force_length(text, req_length):
        len_diff = len(text) - req_length

        if len_diff > 0:
            return text[:req_length - 3] + "..."
        elif len_diff < 0:
            return text + " " * abs(len_diff)
        else:
            return text

    @staticmethod
    def print_log(tag, msg, log_level, prefix="", suffix="\n"):
        if log_level.value > POILog.MAX_LOG_LEVEL.value:
            return

        time = datetime.datetime.now().strftime("%H:%M:%S")
        head = POILog.apply_style(f" POILog | {time} ", POILog.color_codes["bw"])
        formatted_tag = POILog.force_length(tag, POILog.TAG_COLUMN_WIDTH)
        color = POILog.color_map[log_level]
        colored_log_level = POILog.apply_style(str(log_level.name), color)
        formatted_log_level = POILog.force_length(f"[{colored_log_level}]", POILog.LEVEL_COLUMN_WIDTH)

        msg_lines = msg.split("\n")

        info_text = f"{prefix}{head}  {formatted_tag}  {formatted_log_level}  "

        len_codes = sum([info_text.count(code) * len(code) for code in POILog.color_codes.values()])
        info_len = len(info_text) - len_codes

        for idx, msg_line in enumerate(msg_lines):
            if idx == (len(msg_lines) - 1):
                end = suffix
            else:
                end = "\n"

            if idx == 0:
                pre_msg = info_text
            else:
                pre_msg = " " * info_len

            print(f"{pre_msg}{msg_line}", end=end)

    @classmethod
    def e(cls, tag, msg, prefix="", suffix="\n"):
        cls.print_log(tag, msg, LogLevel.ERROR, prefix=prefix, suffix=suffix)

    @classmethod
    def w(cls, tag, msg, prefix="", suffix="\n"):
        cls.print_log(tag, msg, LogLevel.WARN, prefix=prefix, suffix=suffix)

    @classmethod
    def i(cls, tag, msg, prefix="", suffix="\n"):
        cls.print_log(tag, msg, LogLevel.INFO, prefix=prefix, suffix=suffix)

    @classmethod
    def d(cls, tag, msg, prefix="", suffix="\n"):
        cls.print_log(tag, msg, LogLevel.DEBUG, prefix=prefix, suffix=suffix)

    @classmethod
    def v(cls, tag, msg, prefix="", suffix="\n"):
        cls.print_log(tag, msg, LogLevel.VERBOSE, prefix=prefix, suffix=suffix)

    @classmethod
    def log_on_exception(cls, tag, msg, exceptions_manager, block):
        try:
            block()
        except Exception as ex:
            protocol = exceptions_manager.apply_protocol(ex)
            cls.e(tag, msg)
            protocol()

    @classmethod
    async def log_on_exception_blocking(cls, tag, msg, exceptions_manager, block):
        try:
            await block()
        except Exception as ex:
            protocol = exceptions_manager.apply_protocol(ex)
            cls.e(tag, msg)
            protocol()

    @staticmethod
    def linify(text):
        return text.replace("\t", "").replace("\n", " ")


class ExceptionsManager:
    def __init__(self, protocols):
        self.protocols = protocols

    def apply_protocol(self, ex):
        return self.protocols.get(type(ex), lambda: None)


def main():
    POILog.i("TEST", "Prova0")
    POILog.i("TEST", "Prova1\nProva2\nProva3\nProva4", suffix=" -> FINE\n")
    POILog.i("TEST", "Prova5")


if __name__ == "__main__":
    main()
