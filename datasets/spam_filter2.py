import pandas as pd
import re
from typing import Dict, List, Tuple
from datetime import datetime
import argparse
import sys

class ReviewSpamFilter:
    """
    A comprehensive spam filter for customer reviews that detects:
    - Links and promotional content
    - Reviews below minimum word/character thresholds
    - Duplicate reviews across different users
    - Suspicious patterns (excessive caps, punctuation)
    """
    
    def __init__(self, 
                 min_words: int = 10,
                 min_chars: int = 50,
                 link_pattern: str = r'(https?://|www\.|bit\.ly|tinyurl)',
                 promo_pattern: str = r'(discount|promo|code|coupon|% off|save \$)'):
        """
        Initialize the spam filter with configurable parameters.
        
        Args:
            min_words: Minimum word count for legitimate reviews
            min_chars: Minimum character count for legitimate reviews
            link_pattern: Regex pattern for detecting links
            promo_pattern: Regex pattern for detecting promotional content
        """
        self.min_words = min_words
        self.min_chars = min_chars
        self.link_pattern = re.compile(link_pattern, re.IGNORECASE)
        self.promo_pattern = re.compile(promo_pattern, re.IGNORECASE)
        self.review_hashes = {}  # HashMap for duplicate detection
        
    def process_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process reviews and identify spam based on multiple criteria.
        
        Args:
            df: DataFrame with columns: business_name, author_name, text, rating, etc.
            
        Returns:
            DataFrame with additional columns: is_spam, spam_reasons, word_count, char_count
        """
        # Reset hash map for new batch
        self.review_hashes.clear()
        
        # Process each review
        results = []
        for idx, row in df.iterrows():
            spam_analysis = self._analyze_review(row, idx)
            results.append({**row.to_dict(), **spam_analysis})
        
        return pd.DataFrame(results)
    
    def _analyze_review(self, review: pd.Series, index: int) -> Dict:
        """
        Analyze a single review for spam indicators.
        
        Args:
            review: Series containing review data
            index: Index of the review in the dataset
            
        Returns:
            Dictionary with spam analysis results
        """
        spam_reasons = []
        is_spam = False
        
        # Get review text and business name
        review_text = str(review.get('text', '')).strip()
        business_name = str(review.get('business_name', '')).strip()
        
        # Calculate word and character counts
        words = review_text.split()
        word_count = len([w for w in words if w])  # Filter empty strings
        char_count = len(review_text)
        
        # Check minimum word count
        if word_count < self.min_words:
            is_spam = True
            spam_reasons.append(f"Too few words ({word_count} < {self.min_words})")
        
        # Check minimum character count
        if char_count < self.min_chars:
            is_spam = True
            spam_reasons.append(f"Too few characters ({char_count} < {self.min_chars})")
        
        # Check for links
        if self.link_pattern.search(review_text):
            is_spam = True
            spam_reasons.append("Contains links")
        
        # Check for promotional content
        if self.promo_pattern.search(review_text):
            is_spam = True
            spam_reasons.append("Promotional content detected")
        
        # Check for duplicates using hash
        review_hash = f"{business_name}:{review_text.lower()}"
        if review_hash in self.review_hashes:
            is_spam = True
            original_idx = self.review_hashes[review_hash]
            spam_reasons.append(f"Duplicate of review #{original_idx + 1}")
        else:
            self.review_hashes[review_hash] = index
        
        # Check for excessive capitalization
        if len(review_text) > 20:
            uppercase_chars = sum(1 for c in review_text if c.isupper())
            caps_ratio = uppercase_chars / len(review_text) if review_text else 0
            if caps_ratio > 0.7:
                is_spam = True
                spam_reasons.append("Excessive capitalization")
        
        # Check for excessive punctuation
        exclamation_count = review_text.count('!')
        if exclamation_count > 5:
            is_spam = True
            spam_reasons.append("Excessive punctuation")
        
        return {
            'is_spam': is_spam,
            'spam_reasons': '; '.join(spam_reasons) if spam_reasons else '',
            'word_count': word_count,
            'char_count': char_count
        }
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate spam detection statistics.
        
        Args:
            df: Processed DataFrame with spam detection results
            
        Returns:
            Dictionary with statistics
        """
        total_reviews = len(df)
        spam_count = df['is_spam'].sum()
        legitimate_count = total_reviews - spam_count
        spam_rate = (spam_count / total_reviews * 100) if total_reviews > 0 else 0
        
        return {
            'total_reviews': total_reviews,
            'spam_count': spam_count,
            'legitimate_count': legitimate_count,
            'spam_rate': round(spam_rate, 1),
            'avg_word_count': round(df['word_count'].mean(), 1),
            'avg_char_count': round(df['char_count'].mean(), 1)
        }
    
    def export_results(self, df: pd.DataFrame, output_file: str, export_type: str = 'all'):
        """
        Export filtered results to CSV.
        
        Args:
            df: Processed DataFrame
            output_file: Output file path
            export_type: 'all', 'spam', or 'legitimate'
        """
        if export_type == 'spam':
            df_export = df[df['is_spam'] == True].copy()
        elif export_type == 'legitimate':
            df_export = df[df['is_spam'] == False].copy()
        else:
            df_export = df.copy()
        
        df_export.to_csv(output_file, index=False)
        print(f"Exported {len(df_export)} reviews to {output_file}")
    
    def print_summary(self, df: pd.DataFrame, detailed: bool = False):
        """
        Print a summary of spam detection results.
        
        Args:
            df: Processed DataFrame
            detailed: Whether to show detailed spam examples
        """
        stats = self.get_statistics(df)
        
        print("\n" + "="*60)
        print("SPAM DETECTION SUMMARY")
        print("="*60)
        print(f"Total Reviews: {stats['total_reviews']}")
        print(f"Spam Detected: {stats['spam_count']} ({stats['spam_rate']}%)")
        print(f"Legitimate Reviews: {stats['legitimate_count']}")
        print(f"Average Word Count: {stats['avg_word_count']}")
        print(f"Average Character Count: {stats['avg_char_count']}")
        
        if detailed and stats['spam_count'] > 0:
            print("\n" + "-"*60)
            print("SPAM EXAMPLES (First 5):")
            print("-"*60)
            spam_df = df[df['is_spam'] == True].head(5)
            for idx, row in spam_df.iterrows():
                print(f"\nBusiness: {row['business_name']}")
                print(f"Author: {row['author_name']}")
                print(f"Review: {row['text'][:100]}..." if len(str(row['text'])) > 100 else f"Review: {row['text']}")
                print(f"Reasons: {row['spam_reasons']}")


def main():
    """
    Main function to run the spam filter from command line.
    """
    parser = argparse.ArgumentParser(description='Filter spam from customer reviews CSV')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output CSV file path', 
                       default='filtered_reviews.csv')
    parser.add_argument('--min-words', type=int, default=10,
                       help='Minimum word count (default: 10)')
    parser.add_argument('--min-chars', type=int, default=50,
                       help='Minimum character count (default: 50)')
    parser.add_argument('--export-type', choices=['all', 'spam', 'legitimate'],
                       default='all', help='Type of reviews to export')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed spam examples')
    
    args = parser.parse_args()
    
    try:
        # Read CSV file
        print(f"Reading file: {args.input_file}")
        df = pd.read_csv(args.input_file)
        
        # Initialize filter
        filter = ReviewSpamFilter(
            min_words=args.min_words,
            min_chars=args.min_chars
        )
        
        # Process reviews
        print("Processing reviews...")
        df_processed = filter.process_reviews(df)
        
        # Print summary
        filter.print_summary(df_processed, detailed=args.detailed)
        
        # Export results
        if args.output:
            filter.export_results(df_processed, args.output, args.export_type)
        
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)


# Example usage as a module
def example_usage():
    """
    Example of how to use the ReviewSpamFilter class programmatically.
    """
    # Sample data
    data = {
        'business_name': ['Restaurant A', 'Restaurant A', 'Restaurant B', 'Restaurant C'],
        'author_name': ['John', 'Jane', 'Bob', 'Alice'],
        'text': [
            'Great food and service! Would definitely come back.',
            'Great food and service! Would definitely come back.',  # Duplicate
            'Good',  # Too short
            'AMAZING PLACE!!!!!!! CHECK OUT www.promo.com FOR DISCOUNTS!!!'  # Multiple red flags
        ],
        'rating': [5, 4, 3, 5],
        'rating_category': ['Excellent', 'Good', 'Average', 'Excellent']
    }
    
    df = pd.DataFrame(data)
    
    # Initialize and run filter
    spam_filter = ReviewSpamFilter(min_words=5, min_chars=20)
    df_filtered = spam_filter.process_reviews(df)
    
    # Show results
    spam_filter.print_summary(df_filtered, detailed=True)
    
    # Export different types
    spam_filter.export_results(df_filtered, 'all_reviews.csv', 'all')
    spam_filter.export_results(df_filtered, 'spam_only.csv', 'spam')
    spam_filter.export_results(df_filtered, 'legitimate_only.csv', 'legitimate')


if __name__ == "__main__":
    # Uncomment the line below to run the example instead of the main CLI
    # example_usage()
    main()