import sys
import numpy as np
from tensorflow.keras.models import load_model
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder


try:
    import gymnasium as gym
except ModuleNotFoundError:
    print('gymnasium module not found. Try to install with')
    print('pip install gymnasium[box2d]')
    sys.exit(1)




def play(env, model):
    
    seed = 2000
    obs, _ = env.reset(seed=seed)

    # drop initial frames
    action0 = 0
    for i in range(50):
        obs,_,_,_,_ = env.step(action0)
    
    done = False
    frames = []
    # Use VideoRecorder for capturing frames
    video_recorder = VideoRecorder(env, "video/test.mp4", enabled=True)
    while not done:
        p = model.predict(obs.reshape(1,96,96,3)) # adapt to your model
        action = np.argmax(p)  # adapt to your model
        print(action)
        obs, _, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
          # Capture the current frame
        
        # Render the environment and record the frame
        video_recorder.capture_frame()


    # Save the recorded frames as a video
    video_recorder.close()


   



env_arguments = {
    'domain_randomize': False,
    'continuous': False,
    'render_mode': "rgb_array"
}

env_name = 'CarRacing-v2'
env = gym.make(env_name, **env_arguments)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)


# your trained model
model = load_model('models/model1_NoAug_20_epochs.h5')

play(env, model)


