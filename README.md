The environment is created to work with OpenAI Gymnasium. 


When a new object is created, the default arguments will be number of players 2, and render_mode None:

    Number of players can be any int, minimum of 1

    render_mode can be "human" or "rgb_array":

        "human" render mode will render the race in a visual way
        "rgb_array" will run the game only mathematically



The reset method of the object will set the players to the starting possition, and it returns observations and additional info.

    The observations is a dictonary{"agent": agent_location, "target": target_location}:

        both locations are a tuple in the (x, y) shape, in range (0 to WIDTH, 0 to HEIGHT).

    The additional info is a dictionary {"Opponent_{id}": opponent_location}

        the opponent location is a touple in the (x, y) shape, in range (0 to WIDTH, 0 to HEIGHT).



The action method takes an action in the shape (x, y, thrust) and it returns observation, reward, terminated, False and info

    The action must be in range (0 to WIDTH, 0 to HEIGHT, 0 to 101)

    The observations is a dictonary{"agent": agent_location, "target": target_location}:

        both locations are a tuple in the (x, y) shape, in range (0 to WIDTH, 0 to HEIGHT).


    The reward will be a value between -0.05 and 1

        to calculate the reward:
            
            a value -0.05 will be given if the distance did not decrease since last action, 
            a value between 0 and 1 will be calculated depending on the percentage of the distance that was traveled this action
            a value of 1 will be given if the agent gets within 800 units of the target, and a new target will be given


    If one of the palyers completes a full lap (reaches all 4 targets), the game will be terminated.


    The following False value is for gym purposes only and can be ignored


    The additional info is a dictionary {"Opponent_{id}": opponent_location}

        the opponent location is a touple in the (x, y) shape, in range (0 to WIDTH, 0 to HEIGHT).



To create the environemnt the gymnasium library is required

        import gymnasium as gym
    
        env = gym.make("PodRacing-v0", render_mode="human")

    Then the reset and step functions can be called

        observations, info = env.reset()

         observations, reward, terminated, _, info = env.step(action)

