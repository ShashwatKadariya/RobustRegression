import training
import evaluation

def main():
    # Train the models
    print("Training Baseline and Lipschitz models...")
    training.main()

    # Evaluate the models
    print("\nEvaluating models...")
    evaluation.main()

if __name__ == "__main__":
    main()
