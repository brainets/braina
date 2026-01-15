  Use Case: Higher-Order Interactions (HOI)
  Goal: Verify the ability to detect synergistic and redundant interactions in neural data using the hoi toolbox.
  Plan:
   1. Simulation: Generate a synthetic dataset representing 3 signals with specific higher-order dependencies
      (e.g., a "Redundancy" scenario where variables copy a common source, or a "Synergy" scenario like XOR).
   2. Analysis: Use the hoi toolbox (specifically hoi.metrics.Oinfo) to compute the O-information (O-info) for the
      triplet.
   3. Verification: Confirm that the O-info sign correctly reflects the interaction type (Positive = Redundancy,
      Negative = Synergy).
   4. Store the files in the directory /braina/usecases/HOI

