%% CREATE THE MAP
% L-R-U-D-Stay
new_map = zeros(4,4,4);
new_map(1,1,:) = [0,1,0,1];
new_map(1,2,:) = [1,1,0,1];
new_map(1,3,:) = [1,1,0,1];
new_map(1,4,:) = [1,0,0,1];

new_map(2,1,:) = [0,1,1,1];
new_map(2,2,:) = [1,1,1,1];
new_map(2,3,:) = [1,1,1,1];
new_map(2,4,:) = [1,0,1,1];

new_map(3,1,:) = [0,1,1,1];
new_map(3,2,:) = [1,1,1,1];
new_map(3,3,:) = [1,1,1,1];
new_map(3,4,:) = [1,0,1,1];

new_map(4,1,:) = [0,1,1,0];
new_map(4,2,:) = [1,1,1,0];
new_map(4,3,:) = [1,1,1,0];
new_map(4,4,:) = [1,0,1,0];

map_human = new_map;
map_human(:,:,5) = ones(4,4); %can remain still
map_police = new_map;
map_police(:,:,5) = zeros(4,4); %cannot remain still 
N_CELLS_MAP = 16;
discount_factor = 0.8;
Q_function = zeros(N_CELLS_MAP*N_CELLS_MAP, 5, 2); % 5 because the number of actions to be taken
coordinates_states = [];
for i=1:size(map_human,1)
    for j=1:size(map_human,2)
        for k=1:size(map_police,1)
            for l=1:size(map_police,2)
                coordinates_states = [coordinates_states ; [i,j,k,l]];      
            end
        end
    end
end

ITERATIONS = 10000000;
state = find(sum([1,1,4,4]==coordinates_states,2)==4); % number of the initial state
state_inicial = state;
actions = [0,0,-1,1,0 ; -1,1,0,0,0];
Q_first_state_Qlearn = zeros(1,ITERATIONS);

for counter=1:ITERATIONS
    mov = map_human(coordinates_states(state,1), coordinates_states(state,2), :); %choose randomly among movements that can make
    vector = find(mov); % check all movements that are possible
    idx = randperm(length(vector),1);
    choice_mov = vector(idx);
    mov_police = map_police(coordinates_states(state,3), coordinates_states(state,4), :); %choose randomly among movements that can make
    vector_police = find(mov_police); % check all movements that are possible
    idx_police = randperm(length(vector_police),1);
    choice_mov_police = vector_police(idx_police);
    new_coord = [coordinates_states(state,1)+actions(1,choice_mov), coordinates_states(state,2)+actions(2,choice_mov), coordinates_states(state,3)+actions(1,choice_mov_police), coordinates_states(state,4)+actions(2,choice_mov_police)];
    new_state = find(sum(new_coord==coordinates_states,2)==4); % number of the new state
    % TODO: Control how many update for pair (s,a)
    reward = 0;
    if (new_coord(1)==new_coord(3)) && (new_coord(2)==new_coord(4)) % same state
       reward = -10;
    elseif (new_coord(1) == 2) && (new_coord(2) == 2)
       reward = 1;
    end
    Q_function(state,choice_mov,2) = Q_function(state,choice_mov,2)+1; % check the times we have computed this (s,a)
    learning_rate = 1/(Q_function(state,choice_mov,2)^(2/3));
    Q_function(state,choice_mov,1) = (1-learning_rate)*Q_function(state,choice_mov,1)+learning_rate*(reward+discount_factor*max(Q_function(new_state,:,1)));
    Q_first_state_Qlearn(counter) = max(Q_function(state_inicial,:,1));
    
    state = new_state;
end
save Q_first_state_Qlearn
figure;
plot(Q_first_state_Qlearn)
title('Value function initial step')
xlabel('nº simulated (s,a)')
ylabel('V(s)')
