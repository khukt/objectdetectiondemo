import random
import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Constants
FACTORY_WIDTH = 100
FACTORY_HEIGHT = 100

# Utility functions
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_snr(distance, transmit_power, noise_power):
    path_loss_exponent = 2.0
    reference_distance = 1.0
    path_loss = 10 * path_loss_exponent * np.log10(distance / reference_distance)
    snr = transmit_power - path_loss - noise_power
    return snr

def calculate_sinr(snr, interference):
    sinr = snr - interference
    return sinr

class SemanticEncoder:
    def encode(self, message, context):
        return f"{message} | Context: {context}"

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
    def __init__(self, x, y, channels, ai_model, max_bandwidth, max_computation):
        self.x = x
        self.y = y
        self.channels = channels
        self.allocated_channels = {}
        self.ai_model = ai_model
        self.latency = []
        self.reliability = []
        self.resource_utilization = []
        self.task_completion_time = []
        self.max_bandwidth = max_bandwidth
        self.max_computation = max_computation
        self.available_bandwidth = max_bandwidth
        self.available_computation = max_computation
        self.slices = {
            'eMBB': {'bandwidth': max_bandwidth * 0.6, 'computation': max_computation * 0.6},
            'URLLC': {'bandwidth': max_bandwidth * 0.3, 'computation': max_computation * 0.3},
            'mMTC': {'bandwidth': max_bandwidth * 0.1, 'computation': max_computation * 0.1}
        }

    def allocate_channel(self, device, message, context, bandwidth_required, computation_required, transmit_power, noise_power):
        priority = self.ai_model.predict_priority(context)
        available_channels = [ch for ch in self.channels if ch not in self.allocated_channels.values()]
        distance = calculate_distance(self.x, self.y, device.x, device.y)
        snr = calculate_snr(distance, transmit_power, noise_power)
        interference = sum(calculate_snr(calculate_distance(self.x, self.y, d.x, d.y), transmit_power, noise_power) for d in self.allocated_channels.values())
        sinr = calculate_sinr(snr, interference)
        slice_type = 'eMBB' if 'eMBB' in context else 'URLLC' if 'URLLC' in context else 'mMTC'
        start_time = time.time()
        if available_channels and sinr > 10 and self.slices[slice_type]['bandwidth'] >= bandwidth_required and self.slices[slice_type]['computation'] >= computation_required:
            channel = max(available_channels, key=lambda ch: sinr / priority)
            self.allocated_channels[device.device_id] = device
            self.slices[slice_type]['bandwidth'] -= bandwidth_required
            self.slices[slice_type]['computation'] -= computation_required
            self.ai_model.update_history(device.device_id, message, context)
            end_time = time.time()
            self.latency.append(end_time - start_time)
            self.reliability.append(1)
            self.resource_utilization.append((self.max_bandwidth - self.available_bandwidth) / self.max_bandwidth)
            return channel
        else:
            end_time = time.time()
            self.latency.append(end_time - start_time)
            self.reliability.append(0)
            self.resource_utilization.append((self.max_bandwidth - self.available_bandwidth) / self.max_bandwidth)
            return None

    def release_channel(self, device_id, bandwidth_released, computation_released, slice_type):
        if device_id in self.allocated_channels:
            self.slices[slice_type]['bandwidth'] += bandwidth_released
            self.slices[slice_type]['computation'] += computation_released
            del self.allocated_channels[device_id]

    def simulate_task(self, device, task_duration, use_semantic, bandwidth, computation, slice_type, offloaded):
        if not use_semantic and random.random() < 0.1:
            completion_time = task_duration * 2
            self.reliability[-1] = 0
        else:
            completion_time = task_duration

        start_time = time.time()
        time.sleep(completion_time)
        end_time = time.time()

        total_time = end_time - start_time
        self.task_completion_time.append(total_time)
        self.release_channel(device.device_id, bandwidth, computation, slice_type)
        if offloaded:
            self.slices[slice_type]['computation'] += computation

class Machine:
    LOCAL_COMPUTATION_CAPACITY = 10  # Computation units available locally

    def __init__(self, device_id, x, y, controller, encoder, decoder):
        self.device_id = device_id
        self.x = x
        self.y = y
        self.controller = controller
        self.encoder = encoder
        self.decoder = decoder
        self.channel = None

    def send_message(self, message, context, bandwidth_required, computation_required, transmit_power, noise_power):
        semantic_message = self.encoder.encode(message, context)
        self.channel = self.controller.allocate_channel(self, message, context, bandwidth_required, computation_required, transmit_power, noise_power)
        return semantic_message

    def receive_message(self, semantic_message, use_semantic, bandwidth_required, computation_required, slice_type):
        message, context = self.decoder.decode(semantic_message)
        self.perform_task(message, context, use_semantic, bandwidth_required, computation_required, slice_type)

    def perform_task(self, message, context, use_semantic, bandwidth_required, computation_required, slice_type):
        offloaded = False
        if computation_required > self.LOCAL_COMPUTATION_CAPACITY:
            offloaded = True
            self.controller.slices[slice_type]['computation'] -= computation_required
        if self.channel:
            task_duration = self.calculate_task_duration(context, offloaded)
            self.controller.simulate_task(self, task_duration, use_semantic, bandwidth_required, computation_required, slice_type, offloaded)

    def calculate_task_duration(self, context, offloaded):
        if "URLLC" in context:
            return random.uniform(0.1, 0.3) if not offloaded else random.uniform(0.05, 0.15)
        elif "eMBB" in context:
            return random.uniform(0.5, 1.0) if not offloaded else random.uniform(0.25, 0.5)
        elif "mMTC" in context:
            return random.uniform(0.2, 0.5) if not offloaded else random.uniform(0.1, 0.25)
        else:
            return random.uniform(0.5, 1.5) if not offloaded else random.uniform(0.25, 0.75)

def run_simulation(use_semantic):
    encoder = SemanticEncoder()
    decoder = SemanticDecoder()
    ai_model = AIModel()
    controller_x, controller_y = FACTORY_WIDTH / 2, FACTORY_HEIGHT / 2
    available_channels = [1, 2, 3, 4, 5]
    controller = CentralizedController(controller_x, controller_y, available_channels, ai_model, max_bandwidth=100, max_computation=1000)

    machines = [
        Machine("RoboticArm1", random.uniform(0, FACTORY_WIDTH), random.uniform(0, FACTORY_HEIGHT), controller, encoder, decoder),
        Machine("SurveillanceCamera1", random.uniform(0, FACTORY_WIDTH), random.uniform(0, FACTORY_HEIGHT), controller, encoder, decoder),
        Machine("Sensor1", random.uniform(0, FACTORY_WIDTH), random.uniform(0, FACTORY_HEIGHT), controller, encoder, decoder)
    ]
    machine_requirements = {
        "RoboticArm1": ("URLLC, control signal for arm", 1, 10, 20, 'URLLC'),
        "SurveillanceCamera1": ("eMBB, high-definition video feed", 2, 5, 50, 'eMBB'),
        "Sensor1": ("mMTC, environmental sensor data", 3, 1, 5, 'mMTC')
    }
    transmit_power = 23  # in dBm (typical for 5G NR)
    noise_power = -94  # in dBm (typical noise figure)

    for _ in range(30):
        for machine in machines:
            context, _, bandwidth_required, computation_required, slice_type = machine_requirements[machine.device_id]
            if use_semantic:
                semantic_message = machine.send_message("task", context, bandwidth_required, computation_required, transmit_power, noise_power)
                machine.receive_message(semantic_message, use_semantic, bandwidth_required, computation_required, slice_type)
            else:
                message = "task"
                machine.channel = controller.allocate_channel(machine, message, context, bandwidth_required, computation_required, transmit_power, noise_power)
                machine.perform_task(message, context, use_semantic, bandwidth_required, computation_required, slice_type)

    avg_latency = np.mean(controller.latency)
    reliability = np.mean(controller.reliability)
    avg_resource_utilization = np.mean(controller.resource_utilization)
    avg_task_completion_time = np.mean(controller.task_completion_time)

    return avg_latency, reliability, avg_resource_utilization, avg_task_completion_time, controller_x, controller_y, machines

# Streamlit UI
st.title("5G Smart Factory: Semantic Communication vs Traditional Communication with Network Slicing and Offloading")

if st.button("Run Comparison Simulation"):
    st.write("### Running Semantic Communication Simulation...")
    sem_latency, sem_reliability, sem_resource_utilization, sem_task_completion_time, controller_x, controller_y, machines = run_simulation(use_semantic=True)

    st.write("### Running Traditional Communication Simulation...")
    trad_latency, trad_reliability, trad_resource_utilization, trad_task_completion_time, _, _, _ = run_simulation(use_semantic=False)

    st.write("### Comparison Results")
    st.write(f"**Semantic Communication** - Latency: {sem_latency:.4f} s, Reliability: {sem_reliability:.4f}, Resource Utilization: {sem_resource_utilization:.4f}, Task Completion Time: {sem_task_completion_time:.4f} s")
    st.write(f"**Traditional Communication** - Latency: {trad_latency:.4f} s, Reliability: {trad_reliability:.4f}, Resource Utilization: {trad_resource_utilization:.4f}, Task Completion Time: {trad_task_completion_time:.4f} s")

    # Plotting the results for better visualization
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

    # Plot factory layout
    fig, ax = plt.subplots()
    ax.set_xlim(0, FACTORY_WIDTH)
    ax.set_ylim(0, FACTORY_HEIGHT)
    ax.scatter(controller_x, controller_y, c='red', label='Controller (BS)')
    for machine in machines:
        ax.scatter(machine.x, machine.y, label=machine.device_id)
    ax.set_title('Factory Layout')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.legend()
    st.pyplot(fig)
