# Copyright (C) 2024 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from transformers import GPT2Tokenizer, GPT2LMHeadModel


class ChildrenAIAssistant:
    """
    An AI assistant designed to answer children's questions using simple language.
    
    This class uses the GPT-2 model to generate child-friendly responses to questions.
    It ensures responses are simple, clear, and appropriate for young children.
    """

    def __init__(self):
        """Initialize the AI assistant."""
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")

    def generate_response(self, question):
        """
        Generate a child-friendly response to the given question.

        Args:
            question: The question to be answered.

        Returns:
            A child-friendly response to the question.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        # Construct the prompt with instructions
        prompt = '''
        Your task is to answer children's questions using simple language.
        Explain any difficult words in a way a 3-year-old can understand.
        Keep responses under 60 words.
        
        Question:
        '''

        input_text = prompt + question

        # Encode and generate response
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        outputs = self.model.generate(
            inputs,
            max_length=150,
            num_return_sequences=1
        )

        # Decode the response
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer


def main():
    """
    Main function to demonstrate the usage of ChildrenAIAssistant.
    """
    # Initialize the AI assistant
    assistant = ChildrenAIAssistant()
    
    # Example usage
    try:
        question = "Why is the sky blue?"
        response = assistant.generate_response(question)
        print(f"Question: {question}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error generating response: {e}")


if __name__ == "__main__":
    main()