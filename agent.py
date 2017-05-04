import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # Define the number of waypoints states, traffic states and possible actions
        self.possible_waypoints = ['right', 'forward', 'left']
        self.number_of_waypoint_states = len(self.possible_waypoints)
        self.possible_actions = [None, 'right', 'forward', 'left']
        self.number_of_actions = len(self.possible_actions)
        self.possible_light_states = ['green', 'red']
        self.possible_oncoming_states = [None, 'right', 'forward', 'left']
        self.possible_left_states = [None, 'right', 'forward', 'left']
        self.number_of_traffic_states = len(self.possible_light_states) * len(self.possible_oncoming_states) * len(
            self.possible_left_states)
        # The q-matrix whith 96 states and 4 possible actions
        self.q_matrix = np.zeros(
            (self.number_of_traffic_states * self.number_of_waypoint_states, self.number_of_actions))
        # The propability matrix for the next traffic state. 128 possible origins (32 possible traffic states and
        # 4 possible actions taken in each traffic state) crossed with 32 possible outcomes.
        self.p_matrix_traffic = np.zeros(
            (self.number_of_traffic_states * self.number_of_actions, self.number_of_traffic_states)).astype(int)
        # The propability matrix for the next waypoint. 12 possible origins (3 possible given waypoints and 4 possible
        # actions taken) crossed with 3 possible outcomes.
        self.p_matrix_waypoint = np.zeros(
            (self.number_of_waypoint_states * self.number_of_actions, self.number_of_waypoint_states)).astype(int)
        self.restart = True
        # Set the parameters alpha, epsilon and gamma
        self.setInitialParameters()
        # Just for visualization in the console. It stores whether the goal was reached or not.
        self.goal_reached = []
        # Initialize two lists, to keep track of the sum of the positive and negative rewards for each run.
        self.list_of_negative_rewards = []
        # Just for visualization. It counts how often the driving agent takes the best action according to the optimal
        # policy.
        self.optimal_action_taken = [0,0]

    def setInitialParameters(self, alpha=0.6, epsilon=0.2, gamma=0.3):
        """
        Sets the initial parameters alpha, epsilon and gamma
        :param alpha: The initial alpha value
        :param epsilon: The initial epsilon value
        :param gamma: The initial gamma value
        :return: None
        """
        # The learning rate (alpha) starts with 1 and will be decreased over time.
        self.alpha = alpha
        # The discount factor (gamma) is small. Since the next state is pretty much chosen by chance, it makes sense to
        # give more weight to the immediate reward.
        self.gamma = gamma
        # The exploration rate (epsilon) starts with 0.2 and will be decresed over time.
        self.epsilon = epsilon
        # epsilon should be reduced to 0 after 50 steps
        self.epsilon_step = self.epsilon / 50.
        # alpha should be reduced to 0 after 100 steps
        self.alpha_step = self.alpha / 100.

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.restart = True
        # epsilon must be decreased over time. In order to avoid negative numbers resulting from rounding errors with
        # float numbers, an if statement is used
        if self.epsilon > self.epsilon_step:
            self.epsilon -= self.epsilon_step
        else:
            self.epsilon = 0.0
        # alpha should be decreased over time. In order to avoid negative numbers resulting from rounding errors with
        # float numbers, an if statement is used
        if self.alpha > self.alpha_step:
            self.alpha -= self.alpha_step
        else:
            self.alpha = 0.0

        # Create a new entry in the reward lists
        self.list_of_negative_rewards.append(0)

    def getBestAction(self):
        """
        This method looks for the best action in the current state according to the q-matrix. If two or more actions are
        equally good, it will return one if these actions randomly.
        :return: Returns the best action in the current state.
        """
        options = self.q_matrix[self.state]
        indexes = [0];
        max_val = options[0]
        # Find the action(s) with the highest reward.
        for i in range(1, len(options)):
            if options[i] > max_val:
                max_val = options[i]
                indexes = [i]
            elif options[i] == max_val:
                indexes.append(i)
        return random.choice(indexes)

    def getMaxNextStateReward(self, traffic_state, wp_state, action_value):
        """
        This method looks for the maximum possible reward in the next possible state. It will weight the maximum
        possible reward of each possible next state by the probability of each state.
        :param traffic_state: The current traffic state.
        :param wp_state: The current waypoint state.
        :param action_value: The chosen action value.
        :return: The maximum possible reward in the next state weighted by its probability.
        """

        # Calculate the row numbers in the matrixes.
        traffic_row = traffic_state * self.number_of_actions + action_value
        wp_row = wp_state * self.number_of_actions + action_value

        # Get the sum of all outcomes from this state so far.
        sum_traffic_row = np.sum(self.p_matrix_traffic[traffic_row])
        sum_wp_row = np.sum(self.p_matrix_waypoint[wp_row])

        # Build a list of next traffic state propabilities.
        if (sum_traffic_row == 0):
            # This state occured for the first time. We assume each next state will occur with equal propability.
            traffic_props = np.ones(self.number_of_traffic_states) / self.number_of_traffic_states
        else:
            traffic_props = self.p_matrix_traffic[traffic_row] / float(sum_traffic_row)

        # Build a list of next waypoint propabilities.
        if (sum_wp_row == 0):
            # This state occured for the first time. We assume each next state will occur with equal propability.
            wp_props = np.ones(self.number_of_waypoint_states) / self.number_of_waypoint_states
        else:
            wp_props = self.p_matrix_waypoint[wp_row] / float(sum_wp_row)

        # Build a list of the propabilities of all next states.
        props_list = []
        for x in traffic_props:
            for y in wp_props:
                props_list.append(x * y)

        # Calculate the weighted maximum reward.
        max_reward = 0
        for i in range(len(props_list)):
            max_reward += props_list[i] * np.max(self.q_matrix[i])

        return max_reward

    def updatePropabilities(self, traffic_state, wp_state):
        """
        Updates the propability matrixes.
        :param traffic_state: The current traffic state.
        :param wp_state: The current waypoint.
        :return: None
        """
        if (self.restart == True):
            # A reset just happened. There is no former state.
            self.restart = False
        else:
            # Increase the counter for the outcome by one in both probability matrixes.
            self.p_matrix_traffic[self.former_traffic_state * self.number_of_actions + self.former_action_value][
                traffic_state] += 1
            self.p_matrix_waypoint[self.former_wp_state * self.number_of_actions + self.former_action_value][
                wp_state] += 1

    def getOptimalAction(self, inputs, nextwaypoint):
        """
        This method is not allowed for the driving agent. It returns the optimal action according to the optimal policy.
        We need this information to compare the driving agent's behavior to the optimal policy.
        :param inputs: The traffic inputs.
        :param nextwaypoint: The next waypoint
        :return: The optimal action.
        """
        traffic_state = [None]
        if inputs['light'] == 'green':
            traffic_state = [None, 'right', 'forward']
            if inputs['oncoming'] != 'right' and inputs['oncoming'] != 'forward':
                traffic_state = [None, 'right', 'forward', 'left']
        elif inputs['left'] != 'forward':
            traffic_state = [None, 'right']
        if nextwaypoint in traffic_state:
            return nextwaypoint
        else:
            return None

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        # Calculate the current traffic_state. The traffic state is the index of the corresponding row of the q-matrix.
        traffic_state = self.possible_light_states.index(inputs['light']) * len(self.possible_oncoming_states) * len(
            self.possible_left_states)
        traffic_state += self.possible_oncoming_states.index(inputs['oncoming']) * len(self.possible_left_states)
        traffic_state += self.possible_left_states.index(inputs['left'])

        # Assign an ID to the current waypoint
        wp_state = self.possible_waypoints.index(self.next_waypoint)

        # Now that the new states are known, we need to update the probability matrixes
        self.updatePropabilities(traffic_state, wp_state)

        # The state is a combination of the traffic_state and the next_waypoint
        self.state = (traffic_state * self.number_of_waypoint_states + wp_state)

        # TODO: Select action according to your policy
        # This is where the exploration rate is applied.
        if random.random() > self.epsilon:
            action_value = self.getBestAction()
        else:
            action_value = random.randint(0, self.number_of_actions - 1)

        # Convert the action value to an action.
        action = self.possible_actions[action_value]

        # Save the traffic state, waypoint state and the action in order to uptate the propability matrixes in the next
        # iteration
        self.former_traffic_state = traffic_state
        self.former_wp_state = wp_state
        self.former_action_value = action_value

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # Get the q-value
        q_value = self.q_matrix[self.state][action_value]
        # Apply the q-learning formula
        self.q_matrix[self.state][action_value] = q_value + self.alpha * (
            reward + self.gamma * self.getMaxNextStateReward(traffic_state, wp_state, action_value) - q_value)

        # Just for later visualization. Tracks whether the goal was reached or not.
        if reward > 9:
            self.goal_reached.append(True)
        elif deadline == 0:
            self.goal_reached.append(False)

        # Just for visualization and to compare the driving agent's behavior to the optimal policy.
        # This comparison will just be made for the last 40 runs. The first 60 turns are just for training.
        if len(self.goal_reached)>60:
            if action == self.getOptimalAction(inputs, self.next_waypoint):
                self.optimal_action_taken[0] += 1
            else:
                self.optimal_action_taken[1] += 1


        # Tracks the negative and positive reward of each run.
        if reward < 0:
            self.list_of_negative_rewards[len(self.list_of_negative_rewards) - 1] += reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs,
                                                                                                    action,
                                                                                                    reward)  # [debug]

    def displayResult(self, alpha, epsilon):
        """
        Plots the result.
        :return: None
        """

        #Plot how many times the driving agent followed the optimal policy
        ind = np.arange(1)
        width = 0.5
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, [self.optimal_action_taken[0]], width, color='g')
        rects2 = ax.bar(ind + width, [self.optimal_action_taken[1]], width, color='r')
        ax.set_title('Optimal actions taken? Score: {}'.format(1.*self.optimal_action_taken[0]/sum(self.optimal_action_taken)))
        ax.legend((rects1[0], rects2[0]), ('Optimal action taken.', 'Wrong action taken'))
        plt.show()

        ## Plot the negative rewards and wheter the goal has been reached.
        ind = np.arange(100)
        width = 1
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, self.list_of_negative_rewards, width, color='r')
        rects2 = ax.bar(ind, self.goal_reached, width, color='g')
        ax.set_xlabel('Turns')
        ax.set_title('alpha={}, gamma={}, epsilon={}, score={}'.format(alpha, self.gamma, epsilon, sum(self.goal_reached)/100.))
        ax.legend((rects1[0], rects2[0]), ('Sum of negative rewards', 'Goal reached'), loc=4)
        plt.show()


def run(alpha=0.6, epsilon=0.2, gamma=0.3):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent

    # Set the initial alpha and gamma
    a.setInitialParameters(alpha, epsilon, gamma)

    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001,
                    display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    #Display the result
    #a.displayResult(alpha, epsilon)

    # Return the result if the this run
    return sum(a.goal_reached) / 100., sum(a.goal_reached[-40:]) / 40., sum(a.list_of_negative_rewards), sum(
        a.list_of_negative_rewards[-40:])


def runWithSeveralParameters(alpha=1.0, epsilon=0.2, gamma=0.1):
    a_step = 0.1
    g_step = 0.1
    # Create the lists of the alpha and gamma values
    alphas = np.arange(0.0 + a_step, 1.0, a_step)
    gammas = np.arange(0.0 + g_step, 1.0, g_step)
    # Create empty lists to store the data
    score_list = []
    score_40_list = []
    negative_rewards_list = []
    negative_rewards_40_list = []
    for a in alphas:
        for g in gammas:
            score_sum = 0.
            score_40_sum = 0.
            negative_rewards_sum = 0.
            negative_rewards_40_sum = 0.
            # Since the results vary, several runs will be done
            runs = 5
            for i in range(runs):
                score, score_40, negative_rewards, negative_rewards_40 = run(a, 0.2, g)
                score_sum += score
                score_40_sum += score_40
                negative_rewards_sum += negative_rewards
                negative_rewards_40_sum += negative_rewards_40
            # After some runs, the average values are stored in the lists
            score_list.append(score_sum / runs)
            score_40_list.append(score_40_sum / runs)
            negative_rewards_list.append(negative_rewards_sum / runs)
            negative_rewards_40_list.append(negative_rewards_40_sum / runs)

        # Print the scores of the gamma runs for the alpha value
        ind = np.arange(len(gammas))
        width = 0.4
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, score_list, width, color='y')
        rects2 = ax.bar(ind + width, score_40_list, width, color='g')
        ax.set_xlabel('gamma')
        ax.set_ylabel('Score')
        ax.set_xticks(ind + width)
        ax.set_xticklabels(gammas)
        ax.set_title('Scores with alpha={}'.format(a))
        ax.legend((rects1[0], rects2[0]), ('Total score', 'Score of last 40 runs'), loc=4)
        plt.show()

        # Print the negative rewards of the gamma runs for the alpha value
        ind = np.arange(len(gammas))
        width = 0.4
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, negative_rewards_list, width, color='r')
        rects2 = ax.bar(ind + width, negative_rewards_40_list, width, color='b')
        ax.set_xlabel('gamma')
        ax.set_ylabel('Negative reward')
        ax.set_xticks(ind + width)
        ax.set_xticklabels(gammas)
        ax.set_title('Negative rewards with alpha={}'.format(a))
        ax.legend((rects1[0], rects2[0]), ('Total negative reward', 'Negative reward of last 40 runs'), loc=4)
        plt.show()

        # Reset lists
        score_list = []
        score_40_list = []
        negative_rewards_list = []
        negative_rewards_40_list = []


if __name__ == '__main__':
    run()
    # runWithSeveralParameters()
