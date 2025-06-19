#
# @lc app=leetcode id=20 lang=python3
#
# [20] Valid Parentheses
#

# @lc code=start
class Solution:
    def isValid(self, s: str) -> bool:
        valid_map = {
            '(': 0,
            '[': 0,
            '{': 0
        }
        
        for char in s:
            if char in valid_map:
                valid_map[char] += 1
            elif char == ')':
                valid_map['('] -= 1
            elif char == ']':
                valid_map['['] -= 1
            elif char == '}':
                valid_map['{'] -= 1
                
            print(f"Current character: {char}, valid_map: {valid_map}")

        # If any count goes negative, it's invalid
        if all(count != 0 for count in valid_map.values()):
            return False
        else:
            return True
# @lc code=end

if __name__ == "__main__":
    sol = Solution()
    print(sol.isValid("([)]"))      # Expected: False