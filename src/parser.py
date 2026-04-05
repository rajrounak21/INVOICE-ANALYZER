import os
import json
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM (Gemini 2.5/3 Flash)
# Ensure GOOGLE_API_KEY is set in .env
llm = init_chat_model(
    model="google_genai:gemini-1.5-flash", 
    api_key=os.getenv("GOOGLE_API_KEY")
)

def clean_json_response(content):
    """
    Cleans the LLM response to ensure it only contains valid JSON.
    Removes markdown code blocks if present.
    """
    content = content.replace("```json", "").replace("```", "").strip()
    return content

def parse_invoice_data(ocr_payload, fields=None):
    """
    Takes OCR payload (Spatial JSON or Tables) and parses it using an LLM.
    
    Args:
        ocr_payload (dict/list): Extracted document data.
        fields (list): Specific fields to extract (e.g. Invoice #, Date, Amount).
    """
    if fields is None:
        fields = ["invoice_number", "date", "vendor_name", "total_amount", "tax_amount", "items"]

    prompt = f"""
    You are an AI Invoice Parsing Specialist.
    Below is extracted OCR data from a document. 
    
    TASK: Extract clinical data including quantities, descriptions, and amounts.
    REQUIRED FIELDS: {", ".join(fields)}
    
    RULES:
    1. If data is missing for a field, return null.
    2. Format output as a JSON object only.
    3. Be precise with numeric values.
    
    OCR DATA:
    {json.dumps(ocr_payload, indent=2)}
    
    JSON RESPONSE:
    """
    
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        content = clean_json_response(response.content)
        return json.loads(content)
    except Exception as e:
        return {"error": str(e), "raw_response": response.content if 'response' in locals() else "FAILED"}

if __name__ == "__main__":
    # Test Parse
    test_data = [{"text": "Invoice #123", "bbox": [10, 10, 50, 20]}, {"text": "Total: $500", "bbox": [10, 50, 50, 60]}]
    result = parse_invoice_data(test_data)
    print("Parsed Data:", json.dumps(result, indent=2))
