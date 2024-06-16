import random
import time
import streamlit as st

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

    def predict_priority(self, message, context):
        if "emergency" in context:
            return 1
        elif "high priority" in context:
            return 2
        else:
            return 3

class CentralizedController:
    def __init__(self, channels, ai_model):
        self.channels = channels
        self.allocated_channels = {}
        self.ai_model = ai_model

    def allocate_channel(self, device_id, message, context):
        priority = self.ai_model.predict_priority(message, context)
        available_channels = [ch for ch in self.channels if ch not in self.allocated_channels.values()]
        if available_channels:
            channel = available_channels[0]
            self.allocated_channels[device_id] = channel
            self.ai_model.update_history(device_id, message, context)
            st.write(f"Channel {channel} allocated to {device_id} for {message} with context {context} and priority {priority}")
            return channel
        else:
            st.write(f"No available channels for {device_id} for {message} with context {context} and priority {priority}")
            return None

    def release_channel(self, device_id):
        if device_id in self.allocated_channels:
            channel = self.allocated_channels.pop(device_id)
            st.write(f"Channel {channel} released by {device_id}")
        else:
            st.write(f"No channel allocated to {device_id}")

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
            st.write(f"{self.device_id} performing task: {message} with context: {context} on channel {self.channel}")
            time.sleep(random.uniform(0.5, 1.5))  # Simulate task duration
            self.controller.release_channel(self.device_id)
        else:
            st.write(f"{self.device_id} cannot perform task: {message}, no channel allocated")

# Streamlit UI
st.title("Smart Factory: Semantic Communication and Wireless Resource Management")

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
machine2 = Machine("ConveyorBelt1", controller, encoder, decoder)
machine3 = Machine("Sensor1", controller, encoder, decoder)

# Simulate the machines sending messages and performing tasks
if st.button("Start Simulation"):
    st.write("### First Cycle")
    semantic_message1 = machine1.send_message("move", "item to arm, high priority")
    machine1.receive_message(semantic_message1)

    semantic_message2 = machine2.send_message("convey", "item to station, normal priority")
    machine2.receive_message(semantic_message2)

    semantic_message3 = machine3.send_message("monitor", "temperature, normal priority")
    machine3.receive_message(semantic_message3)

    st.write("### Second Cycle")
    semantic_message1 = machine1.send_message("move", "item to arm, emergency")
    machine1.receive_message(semantic_message1)

    semantic_message2 = machine2.send_message("convey", "item to station, high priority")
    machine2.receive_message(semantic_message2)

    semantic_message3 = machine3.send_message("monitor", "humidity, normal priority")
    machine3.receive_message(semantic_message3)
