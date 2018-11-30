%% CONSTANTS
REWARD = 1;
N_EPISODES = 500;

%% MAP
% L- R-U-D
new_map = zeros(5,6,4);
new_map(1,1,:) = [0,1,0,1];
new_map(1,3,:) = [0,1,0,1];
new_map(3,5,:) = [0,1,0,1];
new_map(1,4,:) = [1,1,0,1];
new_map(1,5,:) = [1,1,0,1];

new_map(1,6,:) = [1,0,0,1];
new_map(1,2,:) = [1,0,0,1];
new_map(3,6,:) = [1,0,0,1];

new_map(4,2,:) = [1,1,1,0];
new_map(4,3,:) = [1,1,1,0];
new_map(2,6,:) = [1,1,1,0];
new_map(4,4,:) = [1,1,1,0];
new_map(4,5,:) = [1,1,1,0];

new_map(2,1,:) = [0,1,1,1];
new_map(3,1,:) = [0,1,1,1];
new_map(4,1,:) = [0,1,1,1];
new_map(2,3,:) = [0,1,1,1];
new_map(3,3,:) = [0,1,1,1];

new_map(2,2,:) = [1,0,1,1];
new_map(3,2,:) = [1,0,1,1];
new_map(2,4,:) = [1,0,1,1];
new_map(3,4,:) = [1,0,1,1];
new_map(4,6,:) = [1,0,1,1];

new_map(5,1,:) = [0,1,1,0];
new_map(2,5,:) = [0,1,1,0];

new_map(5,5,:) = [0,0,0,0];

new_map(5,2,:) = [1,1,0,0];
new_map(5,3,:) = [1,1,0,0];

new_map(5,4,:) = [1,0,0,0];

new_map(5,6,:) = [1,0,1,0];


new_map(:,:,5) = ones(5,6); % adding the possibility to stay still
%% MAP MINOTAUR
map_minotaur = zeros(5,6,4);

for i=2:4
    for j=2:5
        map_minotaur(i,j,:) = 1;
    end
end
for j = 2:5
    map_minotaur(1,j,:) = [1,1,0,1];
end
for j = 2:5
    map_minotaur(5,j,:) = [1,1,1,0];
end
for i = 2:4
    map_minotaur(i,1,:) = [0,1,1,1];
end
for i = 2:4
    map_minotaur(i,6,:) = [1,0,1,1];
end
% L- R-U-D
map_minotaur(1,1,:) = [0,1,0,1];
map_minotaur(5,6,:) = [1,0,1,0];
map_minotaur(1,6,:) = [1,0,0,1];
map_minotaur(5,1,:) = [0,1,1,0];
%% CHANGE FOR 1.B
map_minotaur(:,:,5) = ones(5,6); % adding the possibility to stay still

%% TRANSITION PROBABILITY NEW
dimension = (size(new_map,1)*size(new_map,2))^2 + 1; %dead state
transition_matrix = zeros(dimension, dimension, 5); %5 for each action (still, left, right, up, down)
coordinates_states = []; %adding dead state, the 4 corresponds to x and y coordinates of minotaur and human
% Do not count when human and minotaur are in the same position as state
for i=1:size(new_map,1)
    for j=1:size(new_map,2)
        for k=1:size(map_minotaur,1)
            for l=1:size(map_minotaur,2)
                coordinates_states = [coordinates_states ; [i,j,k,l]];      
            end
        end
    end
end
coordinates_states = [coordinates_states ; [0,0,0,0]]; % dead state
reward = zeros(size(coordinates_states,1),5); %reward matrix (5 columns for each action), last row always 0 as dead
% loop for transition probability each action
actions = [0,0,-1,1,0 ; -1,1,0,0,0];
for i=1:size(actions,2)
   for j=1:size(coordinates_states,1)-1 % not consider dead state
       if ((coordinates_states(j,1)==coordinates_states(j,3)) && (coordinates_states(j,2)==coordinates_states(j,4))) %transition to dead
           transition_matrix(j,size(coordinates_states,1),i) = 1; % prob 1 of dead state transition
       elseif ((sum(coordinates_states(j,1:2)==[5,5],2)==2) && (sum(coordinates_states(j,3:4)==[5,5],2)~=2)) %add reward
           reward(j,i) = 1;
           transition_matrix(j,size(coordinates_states,1),i) = 1; % prob 1 of dead state transition after winning       
       elseif (new_map(coordinates_states(j,1),coordinates_states(j,2),i)==1) %I can move in that direction
            new_human = [coordinates_states(j,1) + actions(1,i) , coordinates_states(j,2) + actions(2,i)]; %new coordinate human
            new_minotaur = [];
            posibles_mov = find(map_minotaur(coordinates_states(j,3),coordinates_states(j,4),:));
            minotaur_move = coordinates_states(j,3:4).' + actions(:,posibles_mov);
            for counter=1:size(posibles_mov,1) %possible moves
                index = find(sum([new_human, minotaur_move(1,counter), minotaur_move(2,counter)]==coordinates_states,2)==4);
                transition_matrix(j,index,i) = 1/size(posibles_mov,1); %possibility normalize                         
            end
       end   
   end
end
%% VALUE FUNCTION
% position human (1,1) & minotaur (5,5) --> we start from the last
% iteration (T=15)
T_interval = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25];
lambda = 1;
win_average = [];
for ii=1:size(T_interval,2)
    T= T_interval(ii);
    value_function = zeros(size(coordinates_states,1),5,T); % value function in dead always 0, 5 actions human
    value_function(:,:,T) = reward;
    for t=1:T-1
         for action=1:5
             % Hauria devaluar contra la value function de totes les actions
            value_function(:,action,T-t) = reward(:,action) + lambda^(T-t)*max(transition_matrix(:,:,action)*value_function(:,:,T-t+1),[],2);
        end
    end
    %% SIMULATION
    % simulate 100 games
    win_prob = zeros(100,1);
    time = zeros(100,1);

    for game_counter=1:5000
        human = [1,1];
        minotaur = [5,5];
        for t=1:T
            fprintf('Position human: %d,%d \nPosition minotaur: %d,%d \n\n\n',human(1),human(2),minotaur(1),minotaur(2));
            current_state = find(sum([human, minotaur]==coordinates_states,2)==4);
            if (max(value_function(current_state,:,t)))>0
                [~,index_moviment] = max(value_function(current_state,:,t));
            else
                human_pos_movement = find(new_map(human(1), human(2),:));
                index_moviment = human_pos_movement(randi(length(human_pos_movement), 1));
            end
            human = [human(1)+actions(1,index_moviment), human(2)+actions(2,index_moviment)];
            minotaur_pos_movement = find(map_minotaur(minotaur(1), minotaur(2),:));
            movement_index = minotaur_pos_movement(randi(length(minotaur_pos_movement), 1));
            minotaur = [minotaur(1)+actions(1,movement_index), minotaur(2)+actions(2,movement_index)];
            if (sum(human==[5,5],2)==2 && sum(minotaur==[5,5],2)~=2)
                fprintf('Position human: %d,%d \nPosition minotaur: %d,%d \n\n\n',human(1),human(2),minotaur(1),minotaur(2));
                fprintf('WIN\n');
                win_prob(game_counter) = 1;
                time(game_counter) = t;
                break           
            end
            if sum(human==minotaur,2)==2
                fprintf('Position human: %d,%d \nPosition minotaur: %d,%d \n\n\n',human(1),human(2),minotaur(1),minotaur(2));
                fprintf('EATEN\n');
                break           
            end
        end
    end
    win_average = [win_average, mean(win_prob)*100];
end
figure;
plot(T_interval,win_average)
title('Probability win')
xlabel('Life Time T')
ylabel('%')
