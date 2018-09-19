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


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
0
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

        # return Value [-1,1]

        newFood = newFood.asList()
        ghostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
        scared = min(newScaredTimes) > 0

        # if not new ScaredTimes new state is ghost: return lowest value

        if not scared and (newPos in ghostPos):
            return -1.0

        if newPos in currentGameState.getFood().asList():
            return 1

        closestFoodDist = sorted(newFood, key=lambda fDist: util.manhattanDistance(fDist, newPos))
        closestGhostDist = sorted(ghostPos, key=lambda gDist: util.manhattanDistance(gDist, newPos))

        fd = lambda fDis: util.manhattanDistance(fDis, newPos)
        gd = lambda gDis: util.manhattanDistance(gDis, newPos)

        return 1.0 / fd(closestFoodDist[0]) - 1.0 / gd(closestGhostDist[0])

        # return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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

        GhostIndex = [i for i in range(1, gameState.getNumAgents())]

        def term(state, d):
            return state.isWin() or state.isLose() or d == self.depth

        def min_value(state, d, ghost):  # minimizer

            if term(state, d):
                return self.evaluationFunction(state)

            v = 10000000000000000
            for action in state.getLegalActions(ghost):
                if ghost == GhostIndex[-1]:
                    v = min(v, max_value(state.generateSuccessor(ghost, action), d + 1))
                else:
                    v = min(v, min_value(state.generateSuccessor(ghost, action), d, ghost + 1))
            # print(v)
            return v

        def max_value(state, d):  # maximizer

            if term(state, d):
                return self.evaluationFunction(state)

            v = -10000000000000000
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), d, 1))
            # print(v)
            return v

        res = [(action, min_value(gameState.generateSuccessor(0, action), 0, 1)) for action in
               gameState.getLegalActions(0)]
        res.sort(key=lambda k: k[1])

        return res[-1][0]

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        GhostIndex = [i for i in range(1, gameState.getNumAgents())]
        inf = 1e100

        def term(state, d):
            return state.isWin() or state.isLose() or d == self.depth

        def min_value(state, d, ghost, A, B):  # minimizer

            if term(state, d):
                return self.evaluationFunction(state)

            v = inf
            for action in state.getLegalActions(ghost):
                if ghost == GhostIndex[-1]:  # next is maximizer with pacman
                    v = min(v, max_value(state.generateSuccessor(ghost, action), d + 1, A, B))
                else:  # next is minimizer with next-ghost
                    v = min(v, min_value(state.generateSuccessor(ghost, action), d, ghost + 1, A, B))

                if v < A:
                    return v
                B = min(B, v)

            return v

        def max_value(state, d, A, B):  # maximizer

            if term(state, d):
                return self.evaluationFunction(state)

            v = -inf
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), d, 1, A, B))

                if v > B:
                    return v
                A = max(A, v)

            return v

        def alphabeta(state):

            v = -inf
            act = None
            A = -inf
            B = inf

            for action in state.getLegalActions(0):  # maximizing
                tmp = min_value(gameState.generateSuccessor(0, action), 0, 1, A, B)

                if v < tmp:  # same as v = max(v, tmp)
                    v = tmp
                    act = action

                if v > B:  # pruning
                    return v
                A = max(A, tmp)

            return act

        return alphabeta(gameState)

        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        GhostIndex = [i for i in range(1, gameState.getNumAgents())]

        def term(state, d):
            return state.isWin() or state.isLose() or d == self.depth

        def exp_value(state, d, ghost):  # minimizer

            if term(state, d):
                return self.evaluationFunction(state)

            v = 0
            prob = 1 / len(state.getLegalActions(ghost))

            for action in state.getLegalActions(ghost):
                if ghost == GhostIndex[-1]:
                    v += prob * max_value(state.generateSuccessor(ghost, action), d + 1)
                else:
                    v += prob * exp_value(state.generateSuccessor(ghost, action), d, ghost + 1)
            # print(v)
            return v

        def max_value(state, d):  # maximizer

            if term(state, d):
                return self.evaluationFunction(state)

            v = -10000000000000000
            for action in state.getLegalActions(0):
                v = max(v, exp_value(state.generateSuccessor(0, action), d, 1))
            # print(v)
            return v

        res = [(action, exp_value(gameState.generateSuccessor(0, action), 0, 1)) for action in
               gameState.getLegalActions(0)]
        res.sort(key=lambda k: k[1])

        return res[-1][0]

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: based on number 1, added situation values about closestghostdistance, capsules, etc.

    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    walls = currentGameState.getWalls()

    # if not new ScaredTimes new state is ghost: return lowest value

    newFood = newFood.asList()
    ghostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
    scared = min(newScaredTimes) > 0


    if currentGameState.isLose():
        return float('-inf')

    if newPos in ghostPos:
        return float('-inf')


    # if not new ScaredTimes new state is ghost: return lowest value

    closestFoodDist = sorted(newFood, key=lambda fDist: util.manhattanDistance(fDist, newPos))
    closestGhostDist = sorted(ghostPos, key=lambda gDist: util.manhattanDistance(gDist, newPos))

    score = 0

    fd = lambda fDis: util.manhattanDistance(fDis, newPos)
    gd = lambda gDis: util.manhattanDistance(gDis, newPos)

    if gd(closestGhostDist[0]) <3:
        score-=300
    if gd(closestGhostDist[0]) <2:
        score-=1000
    if gd(closestGhostDist[0]) <1:
        return float('-inf')

    if len(currentGameState.getCapsules()) < 2:
        score+=100

    if len(closestFoodDist)==0 or len(closestGhostDist)==0 :
        score += scoreEvaluationFunction(currentGameState) + 10
    else:
        score += (   scoreEvaluationFunction(currentGameState) + 10/fd(closestFoodDist[0]) + 1/gd(closestGhostDist[0]) + 1/gd(closestGhostDist[-1])  )

    return score

    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
