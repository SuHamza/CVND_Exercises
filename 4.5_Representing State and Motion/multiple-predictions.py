# The predict_state function should take in a state
# and a change in time, dt (ex. 3 for 3 seconds)
# and it should output a new, predicted state
# based on a constant motion model
# This function also assumes that all units are in m, m/s, s, etc.
def predict_state(state, dt):
    # Assume that state takes the form [x, vel] i.e. [0, 50]
    
    ## TODO: Calculate the new position, predicted_x
    ## TODO: Calculate the new velocity, predicted_vel
    ## These should be calculated based on the contant motion model:
    ## distance = x + velocity*time
    
    predicted_x = state[0] + (dt * state[1]) 
    predicted_vel = state[1]
    
    # Constructs the predicted state and returns it
    predicted_state = [predicted_x, predicted_vel]
    return predicted_state


## TODO: Click Test Run!

# A state and function call for testing purposes - do not delete
# but feel free to change the values for the test variables
test_state = [10, 3]
test_dt = 5

test_output = predict_state(test_state, test_dt)
print(test_output)

# predict_state takes in a state and a change in time, dt
# So, a call might look like: new_state = predict_state(old_state, 2)

# The car starts at position = 0, going 60 m/s
# The initial state:
initial_state = [10, 60]

# After 2 seconds:
state_est1 = predict_state(initial_state, 2)

# 3 more seconds after the first estimated state
state_est2 = predict_state(state_est1, 3)

## TODO: Use the predict_state function 
## and the above variables to calculate the following states
## (And change their value from 0 to the correct state)

## Then, click Test Run to see your results!

## 1 more second after the second state estimate
state_est3 = predict_state(state_est2, 1)

## 4 more seconds after the third estimated state
state_est4 = predict_state(state_est3, 4)

print(state_est3)
print(state_est4)