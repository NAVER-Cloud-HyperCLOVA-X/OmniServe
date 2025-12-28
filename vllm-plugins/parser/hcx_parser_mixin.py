# HyperCLOVAX vLLM Plugin
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import re

class HcxStreamingParserFunctionsMixin:
    def __init__(self):
        '''
        # initialize these attributes in your class properly
        self.buffer_string = ''
        self.escaped_special_strings = []
        self.special_strings = []
        '''
        pass

    def remove_special_string(self):
        positions = []
        for ss in self.escaped_special_strings:
            positions += [(m.start(), m.end()) for m in re.finditer(ss, self.buffer_string)]

        sorted_positions = sorted(positions, key=lambda x: x[0])
        to_stream = self.buffer_string[:sorted_positions[-1][0]]
        remaining = self.buffer_string[sorted_positions[-1][1]:]
        for ss in self.special_strings:
            to_stream.replace(ss, '')

        return to_stream, remaining


    def check_is_special_string(self):
        for ss in self.special_strings:
            if ss in self.buffer_string:
                return True
        return False

    
    def check_is_part_of_special_string(self):
        for ss in self.special_strings:
            min_len = min(len(self.buffer_string), len(ss))
            for ln in range(min_len, 0, -1):
                if self.buffer_string[-ln:] == ss[:ln]:
                    return True
        return False
