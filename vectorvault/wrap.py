# VECTOR VAULT CONFIDENTIAL
# __________________
# 
#  All Rights Reserved.
# 
# NOTICE:  All information contained herein is, and remains
# the property of Vector Vault and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Vector Vault
# and its suppliers and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Vector Vault.

import textwrap

def wrap(text, max_length=80):
    # Split the text into lines
    lines = text.split('\n')
    
    # Process each line
    for i, line in enumerate(lines):
        if len(line) > max_length:
            # The line is too long - wrap it
            lines[i] = '\n'.join(textwrap.wrap(line, max_length))
    
    # Join the processed lines back together
    return '\n'.join(lines)
