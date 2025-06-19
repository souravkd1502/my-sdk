#
# @lc app=leetcode id=13 lang=python3
#
# [13] Roman to Integer
#

# @lc code=start
class Solution:
    def romanToInt(self, s: str) -> int:
        """
        Convert a Roman numeral string to an integer.
        
        Parameters:
        -----------
        s : str
            A string representing the Roman numeral (e.g., 'MCMXCIV')
        
        Returns:
        --------
        int
            The integer value of the Roman numeral.
        
        Example:
        --------
        >>> roman_to_int('MCMXCIV')
        1994
        """
        # Mapping of single Roman numeral characters to their integer values
        roman_map = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }

        total = 0  # Final result
        prev_value = 0  # Previous numeral value for comparison

        # Process each character from right to left
        for char in reversed(s):
            value = roman_map[char]

            if value < prev_value:
                # Subtract if the current value is less than the previous (e.g., IV)
                total -= value
            else:
                # Otherwise, add the value
                total += value

            # Update previous value
            prev_value = value

        return total
