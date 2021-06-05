
import pulp
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import base64

import fastprogress
import sys
sys.path.append("..")


header=st.beta_container()
dataset=st.beta_container()
features=st.beta_container()
modelTraining=st.beta_container()

@st.cache
def get_data(filename):
	df = pd.read_csv(filename)
	return df



with header:
    from PIL import Image
    img = Image.open("pl_icon.png")
      
    # display image using streamlit
    # width is used to set the width of an image
    st.image(img, width=100)
    st.title('Final Year Project')
    st.write('THIS PROJECT USES LINEAR PROGRAMMING ALGORITHM TO FIND AN OPTIMIZED TEAM TAKING INTO ACCOUNT ALL THE FPL RESTRICTIONS')

# random fake data for costs and values
costs = np.random.uniform(low=5, high=20, size=100)
values = costs * np.random.uniform(low=0.9, high=1.1, size=100)

model = pulp.LpProblem("Constrained value maximisation", pulp.LpMaximize)
decisions = [pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
             for i in range(100)]

# PuLP has a slightly weird syntax, but it's great. This is how to add the objective function:
model += sum(decisions[i] * values[i] for i in range(100)), "Objective"

# and here are the constraints
model += sum(decisions[i] * costs[i] for i in range(100)) <= 100  # total cost
model += sum(decisions) <= 10  # total items

model.solve()


expected_scores = np.random.uniform(low=5, high=20, size=100)
prices = expected_scores * np.random.uniform(low=0.9, high=1.1, size=100)
positions = np.random.randint(1, 5, size=100)
clubs = np.random.randint(0, 20, size=100)

df = get_data('players_raw.csv')

df.sort_values(by=['element_type'])

expected_scores = df["total_points"]  # total points from last season
prices = df["now_cost"] / 10
positions = df["element_type"]
clubs = df["team_code"]
# so we can read the results
fname = df["first_name"]
lname = df["second_name"]
pic = df["code"]



def select_team(expected_scores, prices, positions, clubs, total_budget=100, sub_factor=0.2):
    num_players = len(expected_scores)
    model = pulp.LpProblem("Constrained value maximisation", pulp.LpMaximize)
    decisions = [
        pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    captain_decisions = [
        pulp.LpVariable("y{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    sub_decisions = [
        pulp.LpVariable("z{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]


    # objective function:
    model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * expected_scores[i]
                 for i in range(num_players)), "Objective"

    # cost constraint
    model += sum((decisions[i] + sub_decisions[i]) * prices[i] for i in range(num_players)) <= total_budget  # total cost

    # position constraints
    # 1 starting goalkeeper
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 1) == 1
    # 2 total goalkeepers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 1) == 2

    # 3-5 starting defenders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) <= 5
    # 5 total defenders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 2) == 5

    # 3-5 starting midfielders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) <= 5
    # 5 total midfielders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 3) == 5

    # 1-3 starting attackers
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) >= 1
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) <= 3
    # 3 total attackers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 4) == 3

    # club constraint
    for club_id in np.unique(clubs):
        model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if clubs[i] == club_id) <= 3  # max 3 players

    model += sum(decisions) == 11  # total team size
    model += sum(captain_decisions) == 1  # 1 captain
    
    for i in range(num_players):  
        model += (decisions[i] - captain_decisions[i]) >= 0  # captain must also be on team
        model += (decisions[i] + sub_decisions[i]) <= 1  # subs must not be on team

    model.solve()
    print("Total expected score = {}".format(model.objective.value()))

    return decisions, captain_decisions, sub_decisions


decisions, captain_decisions, sub_decisions = select_team(expected_scores.values, prices.values,
                                                          positions.values, clubs.values,
                                                          sub_factor=0.2)
# print results
p=0
if st.button('CLICK TO GENERATE AN OPTIMIZED TEAM'):
    st.markdown("**PLAYING XI:**")
    st.write("GOALKEEPER:")
    for i in range(df.shape[0]):
        if decisions[i].value() != 0:
            if positions[i] == 1:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]

    st.write("DEFENDERS:")
    for i in range(df.shape[0]):
        if decisions[i].value() != 0:
            if positions[i] == 2:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]

    st.write("MIDFIELDERS:")
    for i in range(df.shape[0]):
        if decisions[i].value() != 0:
            if positions[i] == 3:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]

    st.write("FORWARDS:")
    for i in range(df.shape[0]):
        if decisions[i].value() != 0:
            if positions[i] == 4:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]

    st.markdown("**SUBSTITUTES:**")

    st.write("GOALKEEPER:")
    for i in range(df.shape[0]):
        if sub_decisions[i].value() == 1:

            if positions[i] == 1:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]

    st.write("DEFENDERS:")
    for i in range(df.shape[0]):
        if sub_decisions[i].value() == 1:

            if positions[i] == 2:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]

    st.write("MIDFIELDERS:")
    for i in range(df.shape[0]):
        if sub_decisions[i].value() == 1:

            if positions[i] == 3:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]

    st.write("FORWARDS:")
    for i in range(df.shape[0]):
        if sub_decisions[i].value() == 1:

            if positions[i] == 4:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]


    st.write("Total Cost:")
    st.write("{}".format(p))
import pulp
import numpy as np

position_data = {
    "gk": {"position_id": 1, "min_starters": 1, "max_starters": 1, "num_total": 2},
    "df": {"position_id": 2, "min_starters": 3, "max_starters": 5, "num_total": 5},
    "mf": {"position_id": 3, "min_starters": 3, "max_starters": 5, "num_total": 5},
    "fw": {"position_id": 4, "min_starters": 1, "max_starters": 3, "num_total": 3},
}


def get_decision_array(name, length):
    return np.array([
        pulp.LpVariable("{}_{}".format(name, i), lowBound=0, upBound=1, cat='Integer')
        for i in range(length)
    ])


class TransferOptimiser:
    def __init__(self, expected_scores, buy_prices, sell_prices, positions, clubs):
        self.expected_scores = expected_scores
        self.buy_prices = buy_prices
        self.sell_prices = sell_prices
        self.positions = positions
        self.clubs = clubs
        self.num_players = len(buy_prices)

    def instantiate_decision_arrays(self):
        # we will make transfers in and out of the squad, and then pick subs and captains from that squad
        transfer_in_decisions_free = get_decision_array("transfer_in_free", self.num_players)
        transfer_in_decisions_paid = get_decision_array("transfer_in_paid", self.num_players)
        transfer_out_decisions = get_decision_array("transfer_out_paid", self.num_players)
        # total transfers in will be useful later
        transfer_in_decisions = transfer_in_decisions_free + transfer_in_decisions_paid

        sub_decisions = get_decision_array("subs", self.num_players)
        captain_decisions = get_decision_array("captain", self.num_players)
        return transfer_in_decisions_free, transfer_in_decisions_paid, transfer_out_decisions, transfer_in_decisions, sub_decisions, captain_decisions

    def encode_player_indices(self, indices):
        decisions = np.zeros(self.num_players)
        decisions[indices] = 1
        return decisions

    def apply_transfer_constraints(self, model, transfer_in_decisions_free, transfer_in_decisions,
                                   transfer_out_decisions, budget_now):
        # only 1 free transfer
        model += sum(transfer_in_decisions_free) <= 1

        # budget constraint
        transfer_in_cost = sum(transfer_in_decisions * self.buy_prices)
        transfer_out_cost = sum(transfer_out_decisions * self.sell_prices)
        budget_next_week = budget_now + transfer_out_cost - transfer_in_cost
        model += budget_next_week >= 0


    def solve(self, current_squad_indices, budget_now, sub_factor):
        current_squad_decisions = self.encode_player_indices(current_squad_indices)

        model = pulp.LpProblem("Transfer optimisation", pulp.LpMaximize)
        transfer_in_decisions_free, transfer_in_decisions_paid, transfer_out_decisions, transfer_in_decisions, sub_decisions, captain_decisions = self.instantiate_decision_arrays()

        # calculate new team from current team + transfers
        next_week_squad = current_squad_decisions + transfer_in_decisions - transfer_out_decisions
        starters = next_week_squad - sub_decisions

        # points penalty for additional transfers
        transfer_penalty = sum(transfer_in_decisions_paid) * 4

        self.apply_transfer_constraints(model, transfer_in_decisions_free, transfer_in_decisions,
                                        transfer_out_decisions, budget_now)
        self.apply_formation_constraints(model, squad=next_week_squad, starters=starters,
                                         subs=sub_decisions, captains=captain_decisions)

        # objective function:
        model += self.get_objective(starters, sub_decisions, captain_decisions, sub_factor, transfer_penalty, self.expected_scores), "Objective"
        status = model.solve()

        print("Solver status: {}".format(status))

        return transfer_in_decisions, transfer_out_decisions, starters, sub_decisions, captain_decisions

    def get_objective(self, starters, subs, captains, sub_factor, transfer_penalty, scores):
        starter_points = sum(starters * scores)
        sub_points = sum(subs * scores) * sub_factor
        captain_points = sum(captains * scores)
        return starter_points + sub_points + captain_points - transfer_penalty

    def apply_formation_constraints(self, model, squad, starters, subs, captains):
        for position, data in position_data.items():
            # formation constraints
            model += sum(starter for starter, position in zip(starters, self.positions) if position == data["position_id"]) >= data["min_starters"]
            model += sum(starter for starter, position in zip(starters, self.positions) if position == data["position_id"]) <= data["max_starters"]
            model += sum(selected for selected, position in zip(squad, self.positions) if position == data["position_id"]) == data["num_total"]

        # club constraint
        for club_id in np.unique(self.clubs):
            model += sum(selected for selected, club in zip(squad, self.clubs) if club == club_id) <= 3  # max 3 players

        # total team size
        model += sum(starters) == 11
        model += sum(squad) == 15
        model += sum(captains) == 1

        for i in range(self.num_players):
            model += (starters[i] - captains[i]) >= 0  # captain must also be on team
            model += (starters[i] + subs[i]) <= 1  # subs must not be on team


def get_decision_array_2d(name, n_players, n_weeks):
    return np.array([[
        pulp.LpVariable("{}_{}_w{}".format(name, i, j), lowBound=0, upBound=1, cat='Integer')
        for i in range(n_players)
    ] for j in range(n_weeks)])


class MultiHorizonTransferOptimiser(TransferOptimiser):
    """We now plan transfer decisions over multiple weeks. This means we need a 2d array of expected
    scores (n_players x n_weeks) and 2d arrays of decision variables"""
    def __init__(self, expected_scores, buy_prices, sell_prices, positions, clubs,
                 n_weeks):
        super().__init__(expected_scores, buy_prices, sell_prices, positions, clubs)
        self.num_weeks = n_weeks

    def instantiate_decision_arrays(self):
        # we will make transfers in and out of the squad, and then pick subs and captains from that squad
        transfer_in_decisions_free = get_decision_array_2d("transfer_in_free", self.num_players, self.num_weeks)
        transfer_in_decisions_paid = get_decision_array_2d("transfer_in_paid", self.num_players, self.num_weeks)
        transfer_out_decisions = get_decision_array_2d("transfer_out_paid", self.num_players, self.num_weeks)
        # total transfers in will be useful later
        transfer_in_decisions = [a + b for a, b in zip(transfer_in_decisions_free, transfer_in_decisions_paid)]

        sub_decisions = get_decision_array_2d("subs", self.num_players, self.num_weeks)
        captain_decisions = get_decision_array_2d("captain", self.num_players, self.num_weeks)
        return transfer_in_decisions_free, transfer_in_decisions_paid, transfer_out_decisions, transfer_in_decisions, sub_decisions, captain_decisions

    def solve(self, current_squad_indices, budget_now, sub_factor):
        current_squad_decisions = self.encode_player_indices(current_squad_indices)
        model = pulp.LpProblem("Transfer optimisation", pulp.LpMaximize)
        (transfer_in_decisions_free_all, transfer_in_decisions_paid_all, transfer_out_decisions_all,
         transfer_in_decisions_all, sub_decisions_all, captain_decisions_all) = self.instantiate_decision_arrays()

        total_points = 0
        for w in range(self.num_weeks):
            transfer_in_decisions_free = transfer_in_decisions_free_all[w]
            transfer_in_decisions_paid = transfer_in_decisions_paid_all[w]
            transfer_out_decisions = transfer_out_decisions_all[w]
            transfer_in_decisions = transfer_in_decisions_all[w]
            sub_decisions = sub_decisions_all[w]
            captain_decisions = captain_decisions_all[w]

            # calculate new team from current team + transfers
            next_week_squad = current_squad_decisions + transfer_in_decisions - transfer_out_decisions
            starters = next_week_squad - sub_decisions

            # points penalty for additional transfers
            transfer_penalty = sum(transfer_in_decisions_paid) * 4

            self.apply_transfer_constraints(model, transfer_in_decisions_free, transfer_in_decisions,
                                            transfer_out_decisions, budget_now)
            self.apply_formation_constraints(model, squad=next_week_squad, starters=starters,
                                             subs=sub_decisions, captains=captain_decisions)

            # objective function:
            total_points += self.get_objective(starters, sub_decisions, captain_decisions, sub_factor, transfer_penalty, self.expected_scores[w])
            print(type(total_points))
            current_squad_decisions = next_week_squad

        model += total_points, "Objective"
        model.solve()

        return transfer_in_decisions_all, transfer_out_decisions_all, sub_decisions_all, sub_decisions_all, captain_decisions_all


num_players = 100
current_team_indices = np.random.randint(0, num_players, size=11)  # placeholder
clubs = np.random.randint(0, 20, size=100)  # placeholder
positions = np.random.randint(1, 5, size=100)  # placeholder
expected_scores = np.random.uniform(0, 10, size=100)  # placeholder

#current_sub_indices = np.random.randint(0, num_players, size=4)  # placeholder
#current_captain_indices = current_team_indices[0]  # placeholder

# convert to binary representation
current_team_decisions = np.zeros(num_players) 
current_team_decisions[current_team_indices] = 1
# convert to binary representation
#current_sub_decisions = np.zeros(num_players) 
#current_sub_decisions[current_sub_indices] = 1
# convert to binary representation
#current_captain_decisions = np.zeros(num_players) 
#current_captain_decisions[current_captain_indices] = 1

model = pulp.LpProblem("Transfer optimisation", pulp.LpMaximize)

transfer_in_decisions = [
    pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
    for i in range(num_players)
]
transfer_out_decisions = [
    pulp.LpVariable("y{}".format(i), lowBound=0, upBound=1, cat='Integer')
    for i in range(num_players)
]

next_week_team = [
    current_team_decisions[i] + transfer_in_decisions[i] - transfer_out_decisions[i]
    for i in range(num_players)
]

for i in range(num_players):
    model += next_week_team[i] <= 1
    model += next_week_team[i] >= 0
    model += (transfer_in_decisions[i] + transfer_out_decisions[i]) <= 1
    
# formation constraints
# 1 starting goalkeeper
model += sum(next_week_team[i] for i in range(num_players) if positions[i] == 1) == 1

# 3-5 starting defenders
model += sum(next_week_team[i] for i in range(num_players) if positions[i] == 2) >= 3
model += sum(next_week_team[i] for i in range(num_players) if positions[i] == 2) <= 5

# 3-5 starting midfielders
model += sum(next_week_team[i] for i in range(num_players) if positions[i] == 3) >= 3
model += sum(next_week_team[i] for i in range(num_players) if positions[i] == 3) <= 5

# 1-3 starting attackers
model += sum(next_week_team[i] for i in range(num_players) if positions[i] == 4) >= 1
model += sum(next_week_team[i] for i in range(num_players) if positions[i] == 4) <= 3

# club constraint
for club_id in np.unique(clubs):
    model += sum(next_week_team[i] for i in range(num_players) if clubs[i] == club_id) <= 3  # max 3 players

model += sum(next_week_team) == 11  # total team size


# placeholder budget and prices
budget_now = 0
buy_prices = sell_prices = np.random.uniform(4, 12, size=100)

transfer_in_cost = sum(transfer_in_decisions[i] * buy_prices[i] for i in range(num_players))
transfer_out_cost = sum(transfer_in_decisions[i] * sell_prices[i] for i in range(num_players))

budget_next_week = budget_now + transfer_out_cost - transfer_in_cost
model += budget_next_week >= 0


# objective function:
model += sum((next_week_team[i]) * expected_scores[i]
             for i in range(num_players)), "Objective"



model.solve()
names = df["first_name"] + " " + df["second_name"]



num_players = 100
current_squad_indices = np.random.randint(0, num_players, size=15)
clubs = np.random.randint(0, 20, size=100)
positions = np.random.randint(1, 5, size=100)
expected_scores = np.random.uniform(0, 10, size=100)
current_squad_decisions = np.zeros(num_players) 
current_squad_decisions[current_team_indices] = 1
# placeholder budget and prices
budget_now = 0
buy_prices = sell_prices = np.random.uniform(4, 12, size=100)

opt = TransferOptimiser(expected_scores, buy_prices, sell_prices, positions, clubs)




transfer_in_decisions, transfer_out_decisions, starters, sub_decisions, captain_decisions = opt.solve(current_squad_indices, budget_now, sub_factor=0.2)


for i in range(num_players):
    if transfer_in_decisions[i].value() == 1:
        print("Transferred in: {} {} {}".format(i, buy_prices[i], expected_scores[i]))
    if transfer_out_decisions[i].value() == 1:
        print("Transferred out: {} {} {}".format(i, sell_prices[i], expected_scores[i]))



expected_scores = df["total_points"] / 38  # penalises players who played fewer games
prices = df["now_cost"] / 10
positions = df["element_type"]
clubs = df["team_code"]
# so we can read the results

decisions, captain_decisions, sub_decisions = select_team(expected_scores, prices.values, positions.values, clubs.values)
player_indices = []

print()
print("First Team:")
for i in range(len(decisions)):
    if decisions[i].value() == 1:
        print("{}{}".format(names[i], "*" if captain_decisions[i].value() == 1 else ""), expected_scores[i], prices[i])
        player_indices.append(i)
print()
print("Subs:")
for i in range(len(sub_decisions)):
    if sub_decisions[i].value() == 1:
        print(names[i], expected_scores[i], prices[i])
        player_indices.append(i)



# next week score forecast: start with points-per-game
score_forecast = df["total_points"] / 38
# let's make up a nonsense forecast to add some dynamics -- +1 to Chelsea players
score_forecast.loc[df["team_code"] == 8] += 1
# -1 for Liverpool players
score_forecast.loc[df["team_code"] == 14] -= 1
score_forecast = score_forecast.fillna(0)





opt = TransferOptimiser(score_forecast.values, prices.values, prices.values, positions.values, clubs.values)
transfer_in_decisions, transfer_out_decisions, starters, sub_decisions, captain_decisions = opt.solve(player_indices, budget_now=0, sub_factor=0.2)



if st.button('CLICK TO CHECK IF ANY TRANSFERS ARE NEEDED'):
    for i in range(len(transfer_in_decisions)):
        if transfer_in_decisions[i].value() == 1:
            st.write("TRANSFER IN: {} {} {}".format(names[i], prices[i], score_forecast[i]))
        if transfer_out_decisions[i].value() == 1:
            st.write("TRANSFER OUT: {} {} {}".format(names[i], prices[i], score_forecast[i]))



agree = st.checkbox('DO YOU WANT TO DO THE ABOVE TRANSFERS')

if agree:
    player_indices = []
    print()
    st.markdown("**PLAYING XI:**")
    st.write("GOALKEEPER")
    for i in range(len(starters)):
        if starters[i].value() != 0:
            
            if positions[i] == 1:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)

    st.write("DEFENDERS")
    for i in range(len(starters)):
        if starters[i].value() != 0:
            
            if positions[i] == 2:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)

    st.write("MIDFIELDERS")
    for i in range(len(starters)):
        if starters[i].value() != 0:

            if positions[i] == 3:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))

                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)

    st.write("FORWARDS")
    for i in range(len(starters)):
        if starters[i].value() != 0:

            if positions[i] == 4:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)
                
    print()
    st.markdown("**SUBSTITUTES:**")
    st.write("GOALKEEPER")
    for i in range(len(starters)):
        if sub_decisions[i].value() == 1:
            
            if positions[i] == 1:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)

    st.write("DEFENDERS")
    for i in range(len(starters)):
        if sub_decisions[i].value() == 1:
            
            if positions[i] == 2:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)

    st.write("MIDFIELDERS")
    for i in range(len(starters)):
        if sub_decisions[i].value() == 1:

            if positions[i] == 3:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))

                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)

    st.write("FORWARDS")
    for i in range(len(starters)):
        if sub_decisions[i].value() == 1:
            if positions[i] == 4:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)



st.markdown(
    """<a href="https://www.google.com/">example.com</a>""", unsafe_allow_html=True,
)
st.markdown('''
    <a href="https://www.google.com">
        <img src="https://media.tenor.com/images/ac3316998c5a2958f0ee8dfe577d5281/tenor.gif" />
    </a>''',
    unsafe_allow_html=True
)
   



