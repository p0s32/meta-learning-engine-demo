# run_analysis.py
"""
Standalone script to run the meta-learning engine analysis
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from core_engine import create_meta_learning_system

def main():
    """Run the meta-learning analysis"""
    print("🎯 Running Meta-Learning Engine Analysis...")
    
    try:
        system = create_meta_learning_system()
        result = system.run_full_analysis()
        
        if result['success']:
            print("✅ Analysis completed successfully!")
            print(f"📊 Generated {result['combined_data']['combined_metadata']['total_slots']} slot analyses")
            print("📄 Results written to results/ folder")
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            
        return result
        
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()