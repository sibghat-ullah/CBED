This complete codebase gives you:
✅ Single command training: python main_pipeline.py --stages all --eval
✅ All evaluation metrics from literature (accuracy, macro F1, BLEU, ROUGE, BERTScore)
✅ Attention analysis for your 50-50 problem
✅ Checkpoint management so you never lose progress
✅ Visualization utilities for confusion matrices and training curves
Run python main_pipeline.py --stages all --eval and everything will execute in sequence! Let me know if you need help with the dataset format or any modifications.


# Usage Instructions
1. Setup
bash# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Prepare your dataset as CSV with columns: query, response, intent
# Place at: data/chatbot_dataset.csv
2. Run Full Training Pipeline
bash# Train all stages
python main_pipeline.py --stages all --eval

# Train specific stages
python main_pipeline.py --stages 1,2  # Only stages 1 and 2

# Resume from checkpoint
python main_pipeline.py --stages 2,3 --resume
3. Run Individual Stages
bash# Stage 1: Encoder training
python train_stage1.py

# Stage 2: Fusion training
python train_stage2.py

# Stage 3: Decoder training
python train_stage3.py
4. Evaluate Trained Model
bashpython evaluate.py

