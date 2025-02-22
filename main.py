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


from typing import Optional
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class ChildrenAIAssistant:
    """
    An AI assistant designed to answer children's questions using simple language.
    
    This class uses the GPT-2 model to generate child-friendly responses to questions.
    It ensures responses are simple, clear, and appropriate for young children.
    
    Attributes:
        tokenizer (GPT2Tokenizer): The tokenizer for processing input text.
        model (GPT2LMHeadModel): The GPT-2 language model for generating responses.
    """

    def __init__(self) -> None:
        """Initialize the AI assistant."""
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")

    def generate_response(self, question: str) -> str:
        """
        Generate a child-friendly response to the given question.

        This method processes the input question and generates a response
        that is appropriate for young children, using simple language
        and explanations.

        Args:
            question (str): The question to be answered.

        Returns:
            str: A child-friendly response to the question.

        Raises:
            ValueError: If the question is empty or contains only whitespace.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        # Construct the prompt with instructions
        prompt = """
        Your task is to answer children's questions using simple language.
        Explain any difficult words in a way a 3-year-old can understand.
        Keep responses under 60 words.
        
        Question: {question}
        Answer:""".format(question=question)

        # Encode and generate response
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors='pt'
        )

        # Generate response with the model
        outputs = self.model.generate(
            inputs,
            max_length=150,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode and clean up the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part from the response
        try:
            answer = response.split("Answer:")[-1].strip()
        except IndexError:
            answer = response.strip()

        return answer

    def __str__(self) -> str:
        """Return a string representation of the assistant."""
        return f"ChildrenAIAssistant(model={self.model.__class__.__name__})"
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the assistant."""
        return "ChildrenAIAssistant()"


def main() -> None:
    """
    Main function to demonstrate the usage of ChildrenAIAssistant.
    """
    # Initialize the AI assistant
    assistant = ChildrenAIAssistant()
    
if __name__ == "__main__":
    main()