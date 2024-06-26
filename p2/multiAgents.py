# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        currFoodList = currentGameState.getFood().asList()
        distance = float("-Inf")

        if action == 'Stop':
            return float("-Inf")

        for ghostState in newGhostStates:
            ghostPosition = ghostState.getPosition()
            if ghostPosition == tuple(list(newPos)) and (ghostState.scaredTimer == 0):
                return float("-Inf")

        for food in currFoodList:
            tempDist = (manhattanDistance(list(newPos), food)) * -1
            if (distance < tempDist):
                distance = tempDist

        return distance

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, depth, agent):
            if agent >= gameState.getNumAgents():
                depth += 1
                agent = 0

            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(agent)
            if agent == 0:
                bestAction = None
                bestValue = -float("inf")

                for action in legalActions:
                    successor = gameState.generateSuccessor(agent, action)
                    successorValue = minimax(successor, depth, agent + 1)

                    if isinstance(successorValue, tuple):
                        successorValue = successorValue[1]

                    if successorValue > bestValue:
                        bestValue = successorValue
                        bestAction = action

                return bestAction, bestValue

            else:
                bestAction = None
                bestValue = float("inf")

                for action in legalActions:
                    successor = gameState.generateSuccessor(agent, action)
                    successorValue = minimax(successor, depth, agent + 1)

                    if isinstance(successorValue, tuple):
                        successorValue = successorValue[1]

                    if successorValue < bestValue:
                        bestValue = successorValue
                        bestAction = action

                return bestAction, bestValue

        bestAction, _ = minimax(gameState, 0, 0)
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, depth, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            v = -float("inf")
            best_action = None

            for action in state.getLegalActions(0):
                successor_state = state.generateSuccessor(0, action)
                val, _ = min_value(successor_state, depth, 1, alpha, beta)

                if val > v:
                    v = val
                    best_action = action

                if v > beta:
                    return v, best_action

                alpha = max(alpha, v)

            return v, best_action

        def min_value(state, depth, agent, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            v = float("inf")
            best_action = None

            for action in state.getLegalActions(agent):
                successor_state = state.generateSuccessor(agent, action)

                if agent == gameState.getNumAgents() - 1:
                    val, _ = max_value(successor_state, depth - 1, alpha, beta)
                else:
                    val, _ = min_value(successor_state, depth, agent + 1, alpha, beta)

                if val < v:
                    v = val
                    best_action = action

                if v < alpha:
                    return v, best_action

                beta = min(beta, v)

            return v, best_action

        _, action = max_value(gameState, self.depth, -float("inf"), float("inf"))
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(gameState, depth, agent):
            if agent >= gameState.getNumAgents():
                agent = 0
                depth += 1

            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None

            legalActions = gameState.getLegalActions(agent)

            if agent == 0:
                bestAction = None
                bestValue = -float("inf")

                for action in legalActions:
                    successor = gameState.generateSuccessor(agent, action)
                    successorValue, _ = expectimax(successor, depth, agent + 1)

                    if successorValue > bestValue:
                        bestValue = successorValue
                        bestAction = action

                return bestValue, bestAction

            else: 
                expectedValue = 0

                for action in legalActions:
                    successor = gameState.generateSuccessor(agent, action)
                    successorValue, _ = expectimax(successor, depth, agent + 1)
                    expectedValue += successorValue

                return expectedValue / len(legalActions), None

        bestValue, bestAction = expectimax(gameState, 0, 0)
        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    foodPosition = currentGameState.getFood().asList()
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    unsafeDistance = []
    foodDistance = []

    for food in foodPosition:
        foodPacDist = manhattanDistance(food, pacmanPosition)
        foodDistance.append(foodPacDist)

    if not foodDistance:
        foodDistance.append(0)

    for ghost in ghostStates:
        dist = manhattanDistance(pacmanPosition, ghost.getPosition())

        if dist < 2:
            unsafeDistance.append(dist)

    food_weight = -1.0
    unsafe_weight = 1.0

    food_score = food_weight * min(foodDistance)
    unsafe_score = unsafe_weight * sum(unsafeDistance)

    total_score = currentGameState.getScore() + food_score + unsafe_score

    return total_score

# Abbreviation
better = betterEvaluationFunction
