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
