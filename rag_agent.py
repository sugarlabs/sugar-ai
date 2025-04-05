# Pippy's AI-Coding Assistant
# Uses a model from HuggingFace with optional 4-bit quantization

import os
import re
import argparse
import torch
from transformers import pipeline

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
You are a smart and helpful assistant designed to answer coding questions using the Sugar Learning Platform.
Instructions:
1. You must ONLY use the information from the provided context to answer the question.
2. If the context doesn't fully answer the question, use your general knowledge to provide the most helpful and accurate answer you can."
3. Do NOT mention the context, documents, or how the answer was generated. Just provide the answer naturally and clearly.
4. When possible, prioritize and include any relevant details from the context.
5. Always answer in a concise, accurate, and helpful manner.

Context: {context}
Question: {question}

Answer:
"""

CHILD_FRIENDLY_PROMPT = """
You are a friendly teacher talking to a child aged 3 to 10 years old.

Rewrite the answer below using simple words and short sentences so a young child can understand it.

Include examples if needed. Stay close to the original meaning. Do not add extra commentary or explanation about what you are doing. .

Here is the answer to simplify:
{original_answer}

"""

def trim_incomplete_sentence(text):
    matches = list(re.finditer(r'\.\s', text))
    if matches:
        last_complete = matches[-1].end()
        return text[:last_complete].strip()
    else:
        return text.strip()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def combine_messages(x):
    if hasattr(x, "to_messages"):
        return "\n".join(msg.content for msg in x.to_messages())
    return str(x)

def extract_answer_from_output(outputs):
    generated_text = outputs[0]['generated_text']
    if "Child-friendly answer:" in generated_text:
        return generated_text.split("Here is the answer to simplify:")[-1].strip()
    return generated_text.split("Answer:")[-1].strip()

class RAG_Agent:
    def __init__(self, model="Qwen/Qwen2-1.5B-Instruct", quantize=True):
        self.use_quant = quantize and torch.cuda.is_available()
        self.model_name = model

        if self.use_quant:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            tokenizer = AutoTokenizer.from_pretrained(model)
            model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model = pipeline("text-generation", model=model_obj, tokenizer=tokenizer, max_new_tokens=512, return_full_text=False, do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.2, truncation=True)
            self.simplify_model = self.model
        else:
            self.model = pipeline("text-generation", model=model, max_new_tokens=512, return_full_text=False, do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.2, truncation=True, torch_dtype=torch.float16, device=0 if torch.cuda.is_available() else -1)
            self.simplify_model = self.model

        self.retriever = None
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.child_prompt = ChatPromptTemplate.from_template(CHILD_FRIENDLY_PROMPT)

    def setup_vectorstore(self, file_paths):
        all_documents = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                if file_path.endswith(".pdf"):
                    loader = PyMuPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                documents = loader.load()
                all_documents.extend(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(all_documents, embeddings)
        return vector_store.as_retriever()

    def get_relevant_document(self, query):
        try:
            results = self.retriever.invoke(query)
            print(f"[DEBUG] Retrieved results: {results}")
            if results and len(results) > 0:
                return results[0], 1.0
            return None, 0.0
        except Exception as e:
            print(f"Error in get_relevant_document: {e}")
            return None, 0.0

    def run(self, question):
        try:
            doc_result, _ = self.get_relevant_document(question)

            if doc_result:
                print("[DEBUG] Retrieved document content:\n")
                print(doc_result.page_content)
            else:
                print("[DEBUG] No document retrieved.")

            context = doc_result.page_content if doc_result else ""

            chain_input = {
                "context": context,
                "question": question
            }

            prompt_input = self.prompt.format(**chain_input)
            print("\n[DEBUG] Final prompt sent to model:\n")
            print(prompt_input)

            first_response = extract_answer_from_output(self.model(prompt_input))

            second_chain_input = {"original_answer": first_response}
            prompt_for_simplify = self.child_prompt.format(**second_chain_input)
            print("\n[DEBUG] Simplify prompt:\n")
            print(prompt_for_simplify)

            second_response = extract_answer_from_output(self.simplify_model(prompt_for_simplify))

            final_response = trim_incomplete_sentence(second_response)
            output = final_response.strip()
            if output.startswith("Assistant:"):
                output = output[len("Assistant:"):].strip()
            print(output)
            return output

        except Exception as e:
            print(f"Error in run method: {e}")
            return f"Encountered an error: {e}"

def main():
    parser = argparse.ArgumentParser(description="Pippy's AI-Coding Assistant")
    parser.add_argument('--model', type=str, default='bigscience/bloom-1b1')
    parser.add_argument('--docs', nargs='+', default=[
        './docs/Pygame Documentation.pdf',
        './docs/Python GTK+3 Documentation.pdf',
        './docs/Sugar Toolkit Documentation.pdf'
    ])
    parser.add_argument('--quantize', action='store_true')
    args = parser.parse_args()

    try:
        agent = RAG_Agent(model=args.model, quantize=args.quantize)
        agent.retriever = agent.setup_vectorstore(args.docs)
        while True:
            question = input("Enter your question: ").strip()
            if not question:
                print("Please enter a valid question.")
                continue
            response = agent.run(question)
            print("\nResponse:", response)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
