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

    def allocate_channel(self, device_id, message, context):
        priority = self.ai_model.predict_priority(context)
        available_channels = [ch for ch in self.channels if ch not in self.allocated_channels.values()]
        start_time = time.time()
        if available_channels:
            channel = available_channels[0]
            self.allocated_channels[device_id] = channel
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

    def release_channel(self, device_id):
        if device_id in self.allocated_channels:
            channel = self.allocated_channels.pop(device_id)
            return channel
        else:
            return None

    def simulate_task(self, device_id, task_duration):
        start_time = time.time()
        time.sleep(task_duration)  # Simulate task duration
        end_time = time.time()
        completion_time = end_time - start_time
        self.task_completion_time.append(completion_time)
        self.release_channel(device_id)

class Machine:
    def __init__(self, device_id, controller, encoder, decoder):
        self.device_id = device_id
        self.controller = controller
        self.encoder = encoder
        self.decoder = decoder
        self.channel = None

    def send_message(self, message, context):
        semantic_message = self.encoder.encode(message, context)
        self.channel = self.controller.allocate_channel(self.device_id, message, context)
        return semantic_message

    def receive_message(self, semantic_message):
        message, context = self.decoder.decode(semantic_message)
        self.perform_task(message, context)

    def perform_task(self, message, context):
        if self.channel:
            task_duration = self.calculate_task_duration(context)
            self.controller.simulate_task(self.device_id, task_duration)
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

    # Instantiate machines
    machine1 = Machine("RoboticArm1", controller, encoder, decoder)
    machine2 = Machine("SurveillanceCamera1", controller, encoder, decoder)
    machine3 = Machine("Sensor1", controller, encoder, decoder)

    # Simulate the machines sending messages and performing tasks
    for _ in range(30):  # Simulate 30 cycles
        if use_semantic:
            semantic_message1 = machine1.send_message("move", "URLLC, control signal for arm")
            machine1.receive_message(semantic_message1)

            semantic_message2 = machine2.send_message("stream", "eMBB, high-definition video feed")
            machine2.receive_message(semantic_message2)

            semantic_message3 = machine3.send_message("monitor", "mMTC, environmental sensor data")
            machine3.receive_message(semantic_message3)
        else:
            message1 = "move"
            context1 = "URLLC"
            machine1.channel = controller.allocate_channel(machine1.device_id, message1, context1)
            machine1.perform_task(message1, context1)

            message2 = "stream"
            context2 = "eMBB"
            machine2.channel = controller.allocate_channel(machine2.device_id, message2, context2)
            machine2.perform_task(message2, context2)

            message3 = "monitor"
            context3 = "mMTC"
            machine3.channel = controller.allocate_channel(machine3.device_id, message3, context3)
            machine3.perform_task(message3, context3)

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
