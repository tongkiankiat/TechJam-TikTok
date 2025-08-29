import os
import pandas as pd
from dotenv import load_dotenv
import requests
import json
import logging
from tqdm import tqdm
import time
import random
import math
from datetime import datetime, timedelta

# --- CONFIGURATION ---
# The AI model to use with Ollama
MODEL_NAME = "phi4-reasoning"

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"

# Input CSV file path
INPUT_FILE = "hotels_reviews.csv"  # Change this to your actual file path

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- VIOLATION POLICIES ---
VIOLATION_TYPES = {
    "advertisement": {
        "description": "Reviews containing promotional content or links",
        "example": "Best pizza! Visit www.pizzapromo.com for discounts!"
    },
    "irrelevant_content": {
        "description": "Reviews about unrelated topics, not the location",
        "example": "I love my new phone, but this place is too noisy."
    },
    "rant_without_visit": {
        "description": "Rants/complaints from people who never actually visited",
        "example": "Never been here, but I heard it's terrible."
    }
}

# --- SETUP ---
def setup():
    """Set up logging and check Ollama connectivity."""
    load_dotenv()  # Still load .env in case other variables are needed
    
    print("--- Initializing Irrelevant Reviews Generator ---")
    print(f"Using Ollama model: {MODEL_NAME} via {OLLAMA_BASE_URL}")
    
    # Test Ollama connection
    try:
        test_url = f"{OLLAMA_BASE_URL}/api/tags"
        response = requests.get(test_url, timeout=10)
        response.raise_for_status()
        print("Ollama connection successful.")
        logging.info("Ollama connection verified successfully.")
        return True
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Could not connect to Ollama at {OLLAMA_BASE_URL}")
        print("Please ensure Ollama is running and accessible.")
        return False
    except Exception as e:
        print(f"\nERROR: Failed to verify Ollama connection: {e}")
        return False

def load_dataset():
    """Load and validate the input dataset."""
    try:
        df = pd.read_csv(INPUT_FILE)
        required_columns = ['company_name', 'reviewer_name', 'review_date', 'text', 'stars', 'category', 'image_urls', 'image_captions', 'group']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"ERROR: Missing required columns: {missing_columns}")
            return None
            
        print(f"Dataset loaded successfully: {len(df)} rows, {len(df['company_name'].unique())} unique companies")
        return df
        
    except FileNotFoundError:
        print(f"ERROR: Could not find file '{INPUT_FILE}'")
        return None
    except Exception as e:
        print(f"ERROR: Could not load dataset: {e}")
        return None

def create_prompt(company_name, company_category, violation_type, existing_reviews_sample):
    """Create a prompt for generating an irrelevant review using Phi4."""
    
    system_instructions = """You are a review generator that creates realistic but policy-violating fake reviews.
Your task is to generate exactly ONE review that violates a specific policy.

CRITICAL: Your response must be a valid JSON object with exactly two keys:
- "review_text": the review content as a string
- "star_rating": an integer from 1 to 5 representing the star rating

Example format:
{
  "review_text": "The actual review content goes here...",
  "star_rating": 2
}

Do not include any other text, explanations, or formatting outside the JSON object."""

    base_context = f"""
Company: "{company_name}" (a {company_category} business)

Here are some real reviews for context:
{existing_reviews_sample}

VIOLATION TYPE: {violation_type}
"""

    if violation_type == "advertisement":
        specific_instruction = """
Generate a review that contains promotional content, links, or advertisements. This content should look like spam and DOES NOT have to be related to the company being reviewed. Think of it as a random, unrelated business advertising their own services in the review section.

Include elements like:
- Website URLs or promotional links. CRITICAL: Make these look realistic (e.g., 'www.quick-cash-now.biz' or 'best-crypto-deals.io'), not generic placeholders like 'www.example.com'.
- Phone numbers or contact information. CRITICAL: Make these look realistic (e.g., '(555) 808-9921'), not simple patterns like '123-456-7890'.
- Discount codes or special offers for unrelated products/services.
- Overly promotional language and marketing speak.
- Calls to action for a completely different business.

For star rating: Advertisement reviews typically give high ratings (4-5 stars) to seem positive and promotional, but sometimes use medium ratings (3 stars) to appear more genuine.
"""

    elif violation_type == "irrelevant_content":
        specific_instruction = """
Generate a review that talks about completely unrelated topics instead of the business location/service.

Write about things like:
- Personal life stories unrelated to the business
- Reviews of completely different products/services
- Random thoughts about unrelated topics
- Current events, weather, politics that don't relate to business experience
- Anything that has nothing to do with this company

For star rating: Since the content is irrelevant, the star rating can be random/inconsistent (any rating from 1-5), often not matching the tone of the irrelevant content.
"""

    else:  # rant_without_visit
        specific_instruction = """
Generate a review from someone who clearly has never actually visited this business.
Make it a negative/complaining review but obvious they haven't been there.

Include indicators like:
- "Never been here but..."
- "I heard that..."
- "My friend told me..."
- "People say..."
- Generic complaints that could apply to any business
- Lack of specific details about the actual location/service
- Second-hand information or rumors

For star rating: These are typically very low ratings (1-2 stars) since they're complaints and rants, occasionally 3 stars for "neutral" complaints.
"""

    return f"{system_instructions}\n\n{base_context}\n\n{specific_instruction}\n\nGenerate the JSON response now:"

def generate_fake_reviewer_name():
    """Generate a realistic fake reviewer name."""
    first_names = ["Alex", "Jordan", "Casey", "Riley", "Morgan", "Taylor", "Jamie", "Avery", "Quinn", "Sage", "Sam", "Chris", "Pat", "Dana", "Robin"]
    last_names = ["Smith", "Johnson", "Brown", "Davis", "Wilson", "Miller", "Moore", "Taylor", "Anderson", "Thomas", "Garcia", "Martinez", "Lee", "Walker", "Hall"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def generate_fake_date():
    """Generate a realistic fake review date within the last 2 years."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years ago
    
    random_date = start_date + timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds()))
    )
    return random_date.strftime("%Y-%m-%d")

def generate_review_with_ollama(prompt):
    """
    Uses Ollama API to generate a review with star rating based on the prompt.
    Returns a dictionary with 'review_text' and 'star_rating' or None if an error occurs.
    """
    ollama_url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "format": "json",  # Request JSON format
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}

    try:
        logging.debug(f"Sending prompt to Ollama (model: {MODEL_NAME}) for JSON response")
        
        response = requests.post(ollama_url, data=json.dumps(payload), headers=headers, timeout=60.0)
        response.raise_for_status()

        response_data = response.json()
        ai_raw_output = response_data.get("response")

        if ai_raw_output is None:
            logging.error(f"Ollama response missing 'response' field or it's null. Response data: {response_data}")
            return None
            
        if not ai_raw_output.strip():
            logging.warning(f"Ollama returned an empty response string. Response data: {response_data}")
            return None

        logging.debug(f"Raw AI output from Ollama: {ai_raw_output}")
        
        # Parse JSON response
        try:
            # Try to extract JSON from the response
            json_str = ""
            # Look for JSON in code blocks or direct JSON
            import re
            json_match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", ai_raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1) if json_match.group(1) else json_match.group(2)
            else:
                # Try to find JSON boundaries
                json_start_index = ai_raw_output.find('{')
                json_end_index = ai_raw_output.rfind('}') + 1
                if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                    json_str = ai_raw_output[json_start_index:json_end_index]
                else:
                    logging.warning(f"Could not find JSON object in AI response: {ai_raw_output}")
                    return None

            parsed_response = json.loads(json_str)
            
            # Extract review text and star rating
            review_text = parsed_response.get("review_text")
            star_rating = parsed_response.get("star_rating")
            
            if not review_text:
                logging.warning(f"AI response JSON did not contain 'review_text'. Parsed JSON: {parsed_response}")
                return None
                
            if star_rating is None:
                logging.warning(f"AI response JSON did not contain 'star_rating'. Parsed JSON: {parsed_response}")
                return None
                
            # Validate star rating
            if not isinstance(star_rating, (int, float)):
                logging.warning(f"AI star rating is not a number: {star_rating}. Type: {type(star_rating)}")
                return None
                
            star_rating_int = int(star_rating)
            if star_rating_int < 1 or star_rating_int > 5:
                logging.warning(f"AI star rating {star_rating_int} is out of range (1-5). Clamping to valid range.")
                star_rating_int = max(1, min(5, star_rating_int))
            
            # Clean up the review text
            cleaned_text = str(review_text).strip()
            # Remove surrounding quotes if present
            if (cleaned_text.startswith('"') and cleaned_text.endswith('"')) or \
               (cleaned_text.startswith("'") and cleaned_text.endswith("'")):
                cleaned_text = cleaned_text[1:-1]
            
            logging.debug(f"Generated review: {cleaned_text[:100]}... (Stars: {star_rating_int})")
            return {
                'review_text': cleaned_text,
                'star_rating': star_rating_int
            }
            
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON from Ollama AI response. Extracted string: '{json_str}'. Raw: {ai_raw_output}")
            return None
        except Exception as e:
            logging.error(f"Error processing extracted Ollama AI response: {e}. Raw: {ai_raw_output}")
            return None

    except requests.exceptions.Timeout:
        logging.error(f"Timeout calling Ollama API. Ensure Ollama is running and the model '{MODEL_NAME}' is available.")
        return None
    except requests.exceptions.ConnectionError:
        logging.error(f"Connection error calling Ollama API. Ensure Ollama is running at {OLLAMA_BASE_URL}.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Ollama API: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in generate_review_with_ollama: {e}")
        return None

def generate_irrelevant_reviews(df):
    """Generate irrelevant reviews for each company using Ollama."""
    print("\nGenerating irrelevant reviews with Ollama...")
    
    # Group by company
    company_groups = df.groupby('company_name')
    all_irrelevant_reviews = []
    
    total_companies = len(company_groups)
    num_policies = len(VIOLATION_TYPES)
    
    for company_name, company_data in tqdm(company_groups, desc="Processing companies"):
        # Calculate reviews per policy for this company
        total_reviews_for_company = len(company_data)
        reviews_per_policy = math.ceil(total_reviews_for_company / 2 / num_policies)
        
        # Skip if calculation results in 0 reviews
        if reviews_per_policy == 0:
            print(f"\nSkipping {company_name}: only {total_reviews_for_company} reviews, would generate 0 per policy")
            continue
            
        print(f"\n{company_name}: {total_reviews_for_company} original reviews → {reviews_per_policy} per policy ({reviews_per_policy * num_policies} total)")
        
        # Get company info
        company_category = company_data['category'].iloc[0]
        company_group = company_data['group'].iloc[0]  # Get the original group value
        sample_reviews = company_data['text'].head(3).tolist()
        existing_reviews_sample = "\n".join([f"- {review[:100]}..." for review in sample_reviews])
        
        # Generate reviews for each violation type
        for violation_type in VIOLATION_TYPES.keys():
            for _ in range(reviews_per_policy):
                try:
                    # Create prompt
                    prompt = create_prompt(company_name, company_category, violation_type, existing_reviews_sample)
                    
                    # Generate review using Ollama
                    generated_data = generate_review_with_ollama(prompt)
                    
                    if generated_data is None:
                        print(f"Failed to generate review for {company_name} ({violation_type})")
                        continue
                    
                    # Create review record
                    irrelevant_review = {
                        'company_name': company_name,
                        'reviewer_name': generate_fake_reviewer_name(),
                        'review_date': generate_fake_date(),
                        'text': generated_data['review_text'],
                        'stars': generated_data['star_rating'],  # Use AI-generated star rating
                        'category': company_category,
                        'image_urls': '',  # Empty for generated reviews
                        'image_captions': '',  # Empty for generated reviews
                        'group': company_group,  # Use the same group as original company data
                        'irrelevant': True,  # Boolean flag for irrelevant reviews
                        'violation_type': violation_type  # Track which policy was violated
                    }
                    
                    all_irrelevant_reviews.append(irrelevant_review)
                    
                    # Small delay to avoid overwhelming Ollama
                    time.sleep(0.2)
                    
                except Exception as e:
                    print(f"\nERROR generating review for {company_name} ({violation_type}): {e}")
                    logging.error(f"Error generating review for {company_name} ({violation_type}): {e}")
                    continue
    
    print(f"\nGenerated {len(all_irrelevant_reviews)} irrelevant reviews using Ollama")
    return all_irrelevant_reviews

def main():
    """Main function to run the irrelevant review generation process."""
    if not setup():
        return
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return
    
    # Generate irrelevant reviews using Ollama
    irrelevant_reviews = generate_irrelevant_reviews(df)
    
    if not irrelevant_reviews:
        print("No irrelevant reviews were generated. Exiting.")
        return
    
    # Save results
    try:
        irrelevant_df = pd.DataFrame(irrelevant_reviews)
        
        # Create output filename
        input_basename = os.path.splitext(os.path.basename(INPUT_FILE))[0]
        output_filename = f"{input_basename}_irrelevant.csv"
        
        irrelevant_df.to_csv(output_filename, index=False)
        
        print(f"\n✅ Successfully saved {len(irrelevant_reviews)} irrelevant reviews to '{output_filename}'")
        
        # Print summary statistics
        print("\n--- Generation Summary ---")
        for violation_type in VIOLATION_TYPES.keys():
            count = len(irrelevant_df[irrelevant_df['violation_type'] == violation_type])
            print(f"{violation_type.replace('_', ' ').title()}: {count} reviews")
        
        # Print model info
        print(f"\nGenerated using Ollama model: {MODEL_NAME}")
        
    except Exception as e:
        print(f"\nERROR: Could not save the CSV file. Error: {e}")
        logging.error(f"Error saving CSV file: {e}")

if __name__ == "__main__":
    main()