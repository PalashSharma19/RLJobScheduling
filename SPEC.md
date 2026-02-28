RL Scheduler Project – Official System Specification
1️⃣ Core System Constants

Number of Nodes: 5
Max Capacity per Node: 10
Task Size Range: 1–3
Total Timesteps per Simulation: 1000

Normal Arrival Probability: 0.7
Burst Window: 200–300
Burst Arrival Probability: 0.95

Load Decrease Rule:
Deterministic -1 per timestep if current_load > 0

Latency Formula:
Latency = 1 + 0.5 × current_load

Overload Condition:
current_load > 10

2️⃣ State Representation (RL)

State =
(Load1_bucket, Load2_bucket, Load3_bucket, Load4_bucket, Load5_bucket, TaskSize)

Bucket Rules:
0 → load = 0
1 → load = 1–3
2 → load = 4–7
3 → load ≥ 8

3️⃣ Reward Function

Reward =

Latency

(0.3 × LoadVariance)

If overload occurs:
Additional -10 penalty

4️⃣ Evaluation Metrics

Average Latency
Overload Count
Load Variance

5️⃣ Stability Freeze

The following must not change without agreement:

Latency formula

Reward structure

State buckets

Task arrival probabilities

Max capacity

Node count
