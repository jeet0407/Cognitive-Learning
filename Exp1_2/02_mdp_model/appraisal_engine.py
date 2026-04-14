from operator import itemgetter


class AppraisalEngine:
    def __init__(self, effort_horizon=20):
        self.effort_horizon = effort_horizon

    @staticmethod
    def _clip_01(value):
        return max(0.0, min(1.0, value))

    def appraise_suddenness(self, t_hat, previous_state, previous_action, current_state):
        if previous_state is None or previous_action is None:
            return 0.0
        state_action = t_hat.get(previous_state, {}).get(previous_action, {})
        total = sum(state_action.values())
        if total <= 0:
            return 0.0
        return 1.0 - (state_action.get(current_state, 0) / total)

    def appraise_goal_relevance(self, td_error):
        return min(1.0, abs(td_error))

    def appraise_conduciveness(self, td_error):
        return max(-1.0, min(1.0, td_error)) / 2.0 + 0.5

    def appraise_power(self, q_table, chosen_state):
        if chosen_state not in q_table or len(q_table[chosen_state]) == 0:
            return 0.0
        q_values = list(q_table[chosen_state].values())
        avg_q = sum(q_values) / len(q_values)
        min_q = min(q_values)
        max_q = max(q_values)
        if abs(min_q) < abs(max_q):
            denom = max_q if max_q != 0 else 1.0
            return abs((max_q - avg_q) / denom)
        denom = min_q if min_q != 0 else -1.0
        return abs((min_q - avg_q) / denom)

    def appraise_urgency(self, step_count, max_steps):
        if max_steps <= 0:
            return 0.0
        return self._clip_01(step_count / max_steps)

    def _goal_states(self, transitions):
        return {state for state in transitions.keys() if str(state).startswith("G")}

    def _policy_action(self, state, chosen_state, chosen_action, q_table, transitions):
        if state == chosen_state and chosen_action in transitions.get(state, {}):
            return chosen_action
        if state in q_table and len(q_table[state]) > 0:
            return max(q_table[state].items(), key=itemgetter(1))[0]
        actions = list(transitions.get(state, {}).keys())
        return actions[0] if actions else None

    def _expected_steps_to_goal(self, start_state, chosen_state, chosen_action, q_table, transitions):
        horizon = self.effort_horizon
        goals = self._goal_states(transitions)
        memo = {}
        visiting = set()

        def dfs(state, depth):
            key = (state, depth)
            if key in memo:
                return memo[key]
            if state in goals:
                memo[key] = 0.0
                return 0.0
            if depth <= 0 or key in visiting:
                return None

            action = self._policy_action(state, chosen_state, chosen_action, q_table, transitions)
            if action is None or action not in transitions.get(state, {}):
                return None

            visiting.add(key)
            expected = 1.0
            for next_state, prob in transitions[state][action].items():
                if prob <= 0:
                    continue
                child = dfs(next_state, depth - 1)
                if child is None:
                    visiting.remove(key)
                    return None
                expected += prob * child
            visiting.remove(key)
            memo[key] = expected
            return expected

        return dfs(start_state, horizon)

    def appraise_effort(self, current_state, chosen_state, chosen_action, q_table, transitions, action_cost=None):
        state = chosen_state if chosen_state is not None else current_state
        if state is None:
            effort = 1.0
        else:
            expected_steps = self._expected_steps_to_goal(state, chosen_state, chosen_action, q_table, transitions)
            if expected_steps is None:
                effort = 1.0
            else:
                effort = self._clip_01(expected_steps / self.effort_horizon)

        if action_cost is None:
            return effort

        return self._clip_01(0.7 * effort + 0.3 * action_cost)

    def compute(self, *, td_error, t_hat, previous_state, previous_action, current_state,
                q_table, chosen_state, chosen_action, step_count, max_steps,
                transitions, action_cost=None):
        suddenness = self.appraise_suddenness(t_hat, previous_state, previous_action, current_state)
        goal_relevance = self.appraise_goal_relevance(td_error)
        conduciveness = self.appraise_conduciveness(td_error)
        power = self.appraise_power(q_table, chosen_state)
        urgency = self.appraise_urgency(step_count, max_steps)
        effort = self.appraise_effort(current_state, chosen_state, chosen_action, q_table, transitions, action_cost)
        return suddenness, goal_relevance, conduciveness, power, urgency, effort
