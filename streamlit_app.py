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
        self.throughput = []
        self.resource_utilization = []

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
        else:
            pass

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
            time.sleep(random.uniform(0.5, 1.5))  # Simulate task duration
            self.controller.release_channel(self.device_id)

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
    for _ in range(10):  # Simulate 10 cycles
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

    return avg_latency, reliability, avg_resource_utilization

# Streamlit UI
st.title("5G Smart Factory: Semantic Communication and Wireless Resource Management")

if st.button("Run Simulation with Semantic Communication"):
    latency, reliability, resource_utilization = run_simulation(use_semantic=True)
    st.write(f"Semantic Communication - Latency: {latency:.4f} s, Reliability: {reliability:.4f}, Resource Utilization: {resource_utilization:.4f}")

if st.button("Run Simulation with Traditional Communication"):
    latency, reliability, resource_utilization = run_simulation(use_semantic=False)
    st.write(f"Traditional Communication - Latency: {latency:.4f} s, Reliability: {reliability:.4f}, Resource Utilization: {resource_utilization:.4f}")
