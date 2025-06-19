#
# @lc app=leetcode id=14 lang=python3
#
# [14] Longest Common Prefix
#

from typing import List


# @lc code=start
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        # Start by assuming the first string is the common prefix
        prefix = strs[0]

        # Compare the prefix with each string in the list
        for string in strs[1:]:
            # Reduce the prefix until it's a prefix of 'string'
            while not string.startswith(prefix):
                prefix = prefix[:-1]  # Trim last character from prefix
                if not prefix:
                    return ""

        return prefix


# @lc code=end