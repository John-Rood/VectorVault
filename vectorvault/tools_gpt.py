from .ai import AI

'''
    ToolsGPT is a set of tools special to large language models. 
    Every tool turns subjectivity into objectivity.
    
    Objectivity is needed for code. 
    Subjectivity is the primary human input.
    These tools bridge the gap between human interaction and code. 
    
    ToolsGPT allows you to get the following for any input:
        1. a Rating - out of 10
        2. a Match - to a list
        3. a Topic - match to a list
        4. Match or Make (M&M) - to list or make new option for the list

        1. Out of 10:
            Useful to get a quality rating
        
        2. Answer Match
            Useful to get an exact match to one option in a set of options
            in: (text and list of answers)
            out: (exact match to one answer in list of answer)

        3. Topic Match
            Useful to classify the topic of conversation
        
        4. Match or Make
            Get a match to a list of options, or make a new one if unrelated
            Useful if you aren't sure if the input will match one of your existing list options, and need flexibility of creating a new one

'''

class ToolsGPT():
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.llm = AI().llm

    def get_rating(self, text: str, concept_to_rate_for: str = 'Quality', model='gpt-3.5-turbo', custom_prompt=False, loop_limit=20) -> int:
        '''
            Get a numeric rating out of 10. Input plain text, and optionally include a concept to rate for
            Concept can be a comparison, or skill, or anything 
            Output is integer
        '''
        prompt_template = custom_prompt if custom_prompt else """Rate the following content 1 - 10. Respond only with a number in integer format.
        Concept to rate out of ten for: {concept_to_rate_for}
        User: Content to classify out of 10: "{content}" \n\n Reponse: Since I am not allowed to give any explination before or after my rating, I'd like to say
        that this number has been carefully considered given your content. My rating out of 10 integer is: """
        prompt = prompt_template.format(concept_to_rate_for=concept_to_rate_for, content=text)

        answer = self.llm(custom_prompt=prompt, model=model).strip().replace('.', '').lower().strip('"').strip("'")
        if self.verbose == True:
            print(f"'Rating Initial Answer: {answer}")

        try:
            answer = int(answer)
        except:
            loops = 0
            while True:
                if loops >= loop_limit:
                    break
                answer = get_number(answer)
                if self.verbose == True:
                    print(f"Rating Answer {loops}: {answer}")
                loops += 1
                try:
                    answer = int(answer)
                    break
                except:
                    pass

        # internal function to objectify
        def get_number(self, text: str, model='gpt-3.5-turbo', custom_prompt=False):
            prompt_template = custom_prompt if custom_prompt else """Respond only with a number in integer format... 
            Example content: 'The number I would give to this is a 7 out of 10.' Example answer: '7' 
            Example content 2: 'This is level 5 of sales quality' Example answer 2: '5' 
            Example content 3: 'This s good, but could be better. I would rate this as an 8 out of 10.' Example answer 3: '8'\n
            User: The following content should be a number between 1 & 10 - Content: "{content}" 
            \n\nResponse: The number is: """
            prompt = prompt_template.format(content=text)

            return self.llm(custom_prompt=prompt, model=model).strip().replace('.', '').lower().strip('"').strip("'")
            
        return answer
        

    def get_yes_no(self, text: str, question: str, model='gpt-3.5-turbo', custom_prompt=False, loop_limit=20) -> str:
        '''
            Get an exact "yes" or "no" to any question, given an input. 
            Be sure to input text to get a yes or no to, then ask the question to answer
        '''
        answer = self.yay_or_nay(text, question, model=model, custom_prompt=custom_prompt)
        if self.verbose == True:
            print(f"Y/N Initial Answer: {answer}")

        if answer == 'yes' or answer == 'no':
            pass
        else:
            loops = 0
            while True:
                if loops >= loop_limit:
                    break
                answer = self.isolate_yes_no(answer)
                if self.verbose == True:
                    print(f"Y/N Answer {loops}: {answer}")
                loops += 1
                if answer == 'yes' or answer == 'no':
                    break
            
        return answer


    def get_match(self, text: str, list_of_options: list, model='gpt-3.5-turbo', custom_prompt=False, loop_limit=20) -> str:
        '''
            This function could be used in a variety of Natural Language Processing (NLP) tasks, 
            such as text classification or intent recognition.

            Classify any text input to a single option contained in a list of options. - Returns exact match to one of items on list.
            Input text, and list of answers: ["list of answers", "is a list of strings", "do not forget"]
        '''
        list_copy = []
        for option in list_of_options:
            list_copy.append(option.strip().replace('.', '').lower().strip('"').strip("'"))
        prompt_template = custom_prompt if custom_prompt else """Respond with one of the options on this list: {list_of_options} 
        Content to classify: "{content}"  \n\n Classifiy the content above into one of the following options: {list_of_options}"""

        prompt = prompt_template.format(content=text, list_of_options=list_copy)

        answer = self.llm(custom_prompt=prompt, model=model).strip().replace('.', '').lower().strip('"').strip("'")
        if self.verbose == True:
            print(f"Get Answer Initial Answer: {answer}")

        loops = 0
        while answer not in list_copy:
            loops += 1
            answer = self.llm(custom_prompt=prompt, model=model).strip().replace('.', '').lower().strip('"').strip("'")
            if self.verbose == True:
                print(f"Answer {loops}: {answer}")
            if loops >= loop_limit:
                break

        answer = list_of_options[list_copy.index(answer)] # return original answer
        
        return answer


    def get_topic(self, text: str, list_of_options: list, model='gpt-3.5-turbo', custom_prompt=False, loop_limit=20) -> str:
        '''
            Like get_match, default optimized for topic recognition
        '''
        list_copy = []
        for option in list_of_options:
            list_copy.append(option.strip().replace('.', '').lower().strip('"').strip("'"))
        prompt_template = custom_prompt if custom_prompt else """Respond with one of the options on this list: {list_of_options} 
        Content to classify: "{content}"  \n\nClassifiy the content above based on which topic it is mostly related to one topic: {list_of_options}"""

        prompt = prompt_template.format(content=text, list_of_options=list_copy)

        topic = self.llm(custom_prompt=prompt, model=model).strip().replace('.', '').lower().strip('"').strip("'")
        if self.verbose == True:
            print(f"Topic Initial Answer: {topic}")

        loops = 0
        while topic not in list_copy:
            loops += 1
            topic = self.llm(custom_prompt=prompt, model=model).strip().replace('.', '').lower().strip('"').strip("'")
            if self.verbose == True:
                print(f"Topic Answer {loops}: {topic}")
            if loops >= loop_limit:
                break
        
        topic = list_of_options[list_copy.index(topic)] # return original topic
        
        return topic


    def match_or_make(self, text, list_of_options: list = [], model='gpt-3.5-turbo', loop_limit=20) -> str:
        ''' 
            "M&M" Returns exact match or new option. Input text and list_of_options. 
            If no list is input, then one will be made and be output

            Internally, decides whether or not to make a new option. 
            Then decides on match is no new option.
            If new option, create it, then finalize it
        '''
        if not list_of_options:
            answer = self.make_option_from_zero(text, list_of_options)
        else:
            yn_prompt = f'''Is this content able to be categorized in one of the provided categories? Categories: {list_of_options} 
                            \nRespond "No" if a new category should be made. 
                            Respond "Yes" if the right category already exists in the list'''
            binary = self.get_yes_no(text, yn_prompt, model=model)
            
            if binary == 'yes':
                answer = self.get_match(text, list_of_options)
            
            if binary == 'no':
                answer = self.make_option(text, list_of_options)

        if self.verbose == True:
            print(f"M&M Initial Answer: {answer}")

        if binary == 'yes':
            return answer
        else:
            loops = 0
            while True:
                if loops >= loop_limit:
                    break
                answer = self.finalize_category(text, answer)
                if self.verbose == True:
                    print(f"M&M Answer {loops}: {answer}")
                loops += 1
                if (len(answer.split())) <= 2:
                    break
        
        return answer
    
    # internal function 
    def make_option(self, text, list_of_options: list, model='gpt-3.5-turbo') -> str:
        prompt_template = """Content to classify: "{content}"  \n\n
        Create a new category for the content based on these other categories in this list: {list_of_options}"""

        prompt = prompt_template.format(content=text, list_of_options=list_of_options)

        return self.llm(custom_prompt=prompt, model=model).strip().replace('.', '').lower().strip('"').strip("'")

    # internal function 
    def make_option_from_zero(self, text, model='gpt-3.5-turbo'):
        '''
            Option without anything to go off of but the text input
        '''
        prompt_template = """Given the following content: {content} 
        Make a simple and general category to classify content like this. Respond only with the category, and 
        no explination before or after, just the name of the catagory. \n\nThe name of the category is: """
        
        prompt = prompt_template.format(content=text)

        return self.llm(custom_prompt=prompt, model=model).strip().replace('.', '').lower().strip('"').strip("'")
    
    # internal function 
    def finalize_category(self, text, prev_answer):
        prompt_template = """Given the following category suggestion: "{prev_answer}"
        \nand this text: "{text}" The final and simple one or two word category name I created for it is: """
        
        prompt = prompt_template.format(prev_answer=prev_answer, text=text)

        return self.llm(custom_prompt=prompt).strip().replace('.', '').lower().strip('"').strip("'")

    # internal function 
    def isolate_yes_no(self, content, question: str, model='gpt-3.5-turbo', custom_prompt=False):
        '''Not recommended for external use. Internal function'''
        prompt_template = custom_prompt if custom_prompt else """Do not respond with anything before the yes or no. Do not add anything after the "yes" or "no". 
        Example question 1: 'Is the Eiffel tower located in Paris?' Example answer 1: 'Yes' Example question 2: 'Do you think I am fat?' 
        Example answer 2: 'No' Example question 3: 'Should I use Matplotlib in Python to draw a graph of csv information I have' Example answer 3: 'Yes'
        Example question 4: 'As an ai language model, I can't tell you yes or no, but I it does have a tendency to work sometimes.' Example answer 4: 'Yes'
        Example question 5: 'Will this happen if it's 19 percent likely to happen?' Example answer 5: 'No'
        \n\nRespond with a "yes" or "no" to following: "{question}"  """

        prompt = prompt_template.format(content=content, question=question)

        return self.llm(custom_prompt=prompt, model=model).strip().replace('.', '').lower().strip('"').strip("'")
    
    def yay_or_nay(self, content, question: str, model='gpt-3.5-turbo', custom_prompt=False):
        '''Not recommended for external use. Internal function'''
        prompt_template = custom_prompt if custom_prompt else """Do not respond with anything before the yes or no. Do not add anything after the "yes" or "no". 
        Example question 1: 'Is the Eiffel tower located in Paris?' Example answer 1: 'Yes' Example question 2: 'Do you think I am fat?' 
        Example answer 2: 'No' Example question 3: 'Should I use Matplotlib in Python to draw a graph of csv information I have' Example answer 3: 'Yes'
        Example question 4: 'As an ai language model, I can't tell you yes or no, but I it does have a tendency to work sometimes.' Example answer 4: 'Yes'
        Example question 5: 'Will this happen if it's 19 percent likely to happen?' Example answer 5: 'No'
        \n\nGiven this content: "{content}", Respond with a "yes" or "no" to following question: "{question}"  """

        prompt = prompt_template.format(content=content, question=question)

        return self.llm(custom_prompt=prompt, model=model).strip().replace('.', '').lower().strip('"').strip("'")