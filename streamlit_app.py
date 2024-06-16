import random
import time
import streamlit as st
import numpy as np

class SemanticEncoder:
    def encode(self, message, context):
        semantic_message = f"{message} | Context: {context}"
        return semantic_message

class SemanticDecoder:
    def decode(self, semantic_message):
        message, context = semantic_message.split(" | Context: ")
        return message, context

class AIModel:
    def __init__(self):
        self.history = []

    def update_history(self, device_id, message, context):
        self.history.append((device_id, message, context))

    def predict_priority(self, context):
        if "URLLC" in context:
            return 1
        elif "eMBB" in context:
            return 2
        elif "mMTC" in context:
            return 3
        else:
            return 4

class CentralizedController:
    def __init__(self, channels, ai_model):
        self.channels = channels
        self.allocated_channels = {}
        self.ai_model = ai_model
        self.latency = []
        self.reliability = []
        self.resource_utilization = []
        self.task_completion_time = []
        self.channel_quality = {ch: random.uniform(0.5, 1.0) for ch in channels}  # Quality between 0.5 to 1.0
        self.bandwidth = {ch: 10 for ch in channels}  # Bandwidth in Mbps
        self.computation_resources = {ch: 100 for ch in channels}  # Computation units

    def allocate_channel(self, device_id, message, context, bandwidth_required, computation_required):
        priority = self.ai_model.predict_priority(context)
        available_channels = [ch for ch in self.channels if ch not in self.allocated_channels.values() and self.bandwidth[ch] >= bandwidth_required and self.computation_resources[ch] >= computation_required]
        start_time = time.time()
        if available_channels:
            # Select channel based on quality and priority
            channel = max(available_channels, key=lambda ch: self.channel_quality[ch] / priority)
            self.allocated_channels[device_id] = channel
            self.bandwidth[channel] -= bandwidth_required
            self.computation_resources[channel] -= computation_required
            self.ai_model.update_history(device_id, message, context)
            end_time = time.time()
            self.latency.append(end_time - start_time)
            self.reliability.append(1)
            self.resource_utilization.append(len(self.allocated_channels) / len(self.channels))
            return channel
        else:
            end_time = time.time()
            self.latency.append(end_time - start_time)
            self.reliability.append(0)
            self.resource_utilization.append(len(self.allocated_channels) / len(self.channels))
            return None

    def release_channel(self, device_id, bandwidth_released, computation_released):
        if device_id in self.allocated_channels:
            channel = self.allocated_channels.pop(device_id)
            self.bandwidth[channel] += bandwidth_released
            self.computation_resources[channel] += computation_released
            return channel
        else:
            return None

    def simulate_task(self, device_id, task_duration, use_semantic, bandwidth, computation):
        # Introduce a failure rate for traditional communication
        if not use_semantic and random.random() < 0.1:  # 10% failure rate
            completion_time = task_duration * 2  # Double the time for recovery
            self.reliability[-1] = 0  # Mark as unreliable
        else:
            completion_time = task_duration
        
        start_time = time.time()
        time.sleep(completion_time)  # Simulate task duration
        end_time = time.time()
        
        total_time = end_time - start_time
        self.task_completion_time.append(total_time)
        self.release_channel(device_id, bandwidth, computation)

class Machine:
    def __init__(self, device_id, controller, encoder, decoder):
        self.device_id = device_id
        self.controller = controller
        self.encoder = encoder
        self.decoder = decoder
        self.channel = None

    def send_message(self, message, context, bandwidth_required, computation_required):
        semantic_message = self.encoder.encode(message, context)
        self.channel = self.controller.allocate_channel(self.device_id, message, context, bandwidth_required, computation_required)
        return semantic_message

    def receive_message(self, semantic_message, use_semantic, bandwidth_required, computation_required):
        message, context = self.decoder.decode(semantic_message)
        self.perform_task(message, context, use_semantic, bandwidth_required, computation_required)

    def perform_task(self, message, context, use_semantic, bandwidth_required, computation_required):
        if self.channel:
            task_duration = self.calculate_task_duration(context)
            self.controller.simulate_task(self.device_id, task_duration, use_semantic, bandwidth_required, computation_required)
        else:
            pass

    def calculate_task_duration(self, context):
        if "URLLC" in context:
            return random.uniform(0.1, 0.3)
        elif "eMBB" in context:
            return random.uniform(0.5, 1.0)
        elif "mMTC" in context:
            return random.uniform(0.2, 0.5)
        else:
            return random.uniform(0.5, 1.5)

def run_simulation(use_semantic):
    # Instantiate the encoder and decoder
    encoder = SemanticEncoder()
    decoder = SemanticDecoder()

    # Instantiate the AI model
    ai_model = AIModel()

    # Instantiate the centralized controller with a list of available channels and the AI model
    available_channels = [1, 2, 3, 4, 5]
    controller = CentralizedController(available_channels, ai_model)

    # Instantiate machines with their required bandwidth and computation resources
    machines = [
        ("RoboticArm1", "URLLC, control signal for arm", 1, 10, 20),  # URLLC
        ("SurveillanceCamera1", "eMBB, high-definition video feed", 2, 5, 50),  # eMBB
        ("Sensor1", "mMTC, environmental sensor data", 3, 1, 5)  # mMTC
    ]

    # Simulate the machines sending messages and performing tasks
    for _ in range(30):  # Simulate 30 cycles
        for device_id, context, _, bandwidth_required, computation_required in machines:
            if use_semantic:
                machine = Machine(device_id, controller, encoder, decoder)
                semantic_message = machine.send_message("task", context, bandwidth_required, computation_required)
                machine.receive_message(semantic_message, use_semantic, bandwidth_required, computation_required)
            else:
                machine = Machine(device_id, controller, encoder, decoder)
                message = "task"
                machine.channel = controller.allocate_channel(device_id, message, context, bandwidth_required, computation_required)
                machine.perform_task(message, context, use_semantic, bandwidth_required, computation_required)

    # Calculate metrics
    avg_latency = np.mean(controller.latency)
    reliability = np.mean(controller.reliability)
    avg_resource_utilization = np.mean(controller.resource_utilization)
    avg_task_completion_time = np.mean(controller.task_completion_time)

    return avg_latency, reliability, avg_resource_utilization, avg_task_completion_time

# Streamlit UI
st.title("5G Smart Factory: Semantic Communication vs Traditional Communication")

if st.button("Run Comparison Simulation"):
    st.write("### Running Semantic Communication Simulation...")
    sem_latency, sem_reliability, sem_resource_utilization, sem_task_completion_time = run_simulation(use_semantic=True)

    st.write("### Running Traditional Communication Simulation...")
    trad_latency, trad_reliability, trad_resource_utilization, trad_task_completion_time = run_simulation(use_semantic=False)

    st.write("### Comparison Results")
    st.write(f"**Semantic Communication** - Latency: {sem_latency:.4f} s, Reliability: {sem_reliability:.4f}, Resource Utilization: {sem_resource_utilization:.4f}, Task Completion Time: {sem_task_completion_time:.4f} s")
    st.write(f"**Traditional Communication** - Latency: {trad_latency:.4f} s, Reliability: {trad_reliability:.4f}, Resource Utilization: {trad_resource_utilization:.4f}, Task Completion Time: {trad_task_completion_time:.4f} s")

    # Plotting the results for better visualization
    import matplotlib.pyplot as plt

    labels = ['Latency (s)', 'Reliability', 'Resource Utilization', 'Task Completion Time (s)']
    semantic_values = [sem_latency, sem_reliability, sem_resource_utilization, sem_task_completion_time]
    traditional_values = [trad_latency, trad_reliability, trad_resource_utilization, trad_task_completion_time]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, semantic_values, width, label='Semantic')
    rects2 = ax.bar(x + width/2, traditional_values, width, label='Traditional')

    ax.set_ylabel('Scores')
    ax.set_title('Semantic vs Traditional Communication')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    st.pyplot(fig)
