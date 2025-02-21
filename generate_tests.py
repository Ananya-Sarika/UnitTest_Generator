import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer,RobertaTokenizer

# Load trained model and tokenizer
model_path = "./trained_model"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer =  RobertaTokenizer.from_pretrained(model_path)

# Function to generate unit tests
def generate_test(function_name, function_code):
    input_text = f"Generate a full Go unit test for the function: {function_name}\n\n{function_code}\n\nInclude assertions and edge cases."

    # Tokenize input
    input_ids = tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).input_ids

    # Generate output
    output_ids = model.generate(
        input_ids, 
        max_length=512, 
        num_beams=5, 
        do_sample=True,  # Ensures variety in generated tests
        temperature=0.7
    )

    # Decode output
    generated_test = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Ensure meaningful output
    if not generated_test.strip():
        return "Error: Model returned an empty output. Try adjusting the prompt or retraining."

    return generated_test


# Example function to test
function_name = "Add"
function_code = """
func Add(a int, b int) int {
    return a+b
}
"""


# Generate test
generated_test_code = generate_test(function_name, function_code)
# tokens = tokenizer.tokenize("""func TestAdd(t *testing.T) { return 1 }
# return nil 
# return 0""")
# print(tokens)

# Print the generated test case
print("Generated Unit Test:\n")
print(generated_test_code)
