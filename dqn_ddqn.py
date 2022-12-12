################################################################################################################
#                                                                                                              #
# Taken from  https://github.com/dongminlee94/deep_rl                                                          #
# Main modified is done by Animesh Kumar Paul (animeshk@ualberta.ca) - Denoted by: AKP_ADDED/AKP_MODIFIED      #
# Thanks Videh Raj Nema (nema@ualberta.ca) for making contribution. His contribution is denoted by: VRN_ADDED  #
# Added tag denotes the newly added codes                                                                      #
################################################################################################################

from common.buffers import *
from common.networks import *
from common.utils import *
import pandas as pd
import torch.optim as optim
from collections import defaultdict



class Agent(object):
    """An implementation of the Deep Q-Network (DQN), Double DQN agents."""

    def __init__(self,
                env,
                args,
                device,
                obs_dim,
                act_num,
                paths,
                run,
                seeds_array_eval,
                steps=0,
                eval_mode=False,
                q_losses=dict(),
                logger=dict(),
    ):
        self.exp = args.exp
        self.env = env
        self.args = args
        self.device = device
        self.obs_dim = obs_dim
        self.act_num = act_num
        self.steps = steps
        self.eval_mode = eval_mode
        self.q_losses = q_losses
        self.logger = logger
        self.nn_params_path, self.data_path, self.tensorboard_path = paths
        self.run = run
        self.seeds_array_eval = seeds_array_eval

        # Main network
        self.qf = MLP(self.obs_dim, self.act_num, hidden_sizes=(self.args.l1_size, self.args.l2_size))
        
        #AKP_ADDED & VJN MODIFY: xavier initialization for main network
        self.qf.apply(init_weights)

        # target network
        self.qf_target = MLP(self.obs_dim, self.act_num, hidden_sizes=(self.args.l1_size, self.args.l2_size))
        # initializing the target network with main network's parameters
        hard_target_update(self.qf, self.qf_target)
        # moving to the device
        self.qf = self.qf.to(device)
        self.qf_target = self.qf_target.to(device)

        # store initial Q-network parameters
        file_path = self.nn_params_path + '/Initial_' + 'Q_Network'+ '_exp_' + str(args.exp) + '_run_'+ str(self.run) + '_replay_frequency_'+ str(args.replay_frequency) + '_steps_' + str(self.steps) + '.pt'
        save_model(self.qf, file_path)
        
        # Create an optimizer
        self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=args.learning_rate, betas=(args.gradient_momentum, args.squared_gradient_momentum))

        # Experience buffer
        self.replay_buffer = ReplayBuffer(self.obs_dim, 1, self.args.replay_capacity, self.device)
        

    def select_action(self, obs):
        # Decaying epsilon
        self.args.epsilon *= self.args.epsilon_decay
        self.args.epsilon = max(self.args.epsilon, self.args.final_epsilon) #AKP_MODIFY

        if np.random.rand() <= self.args.epsilon:
           # Choose a random action with probability epsilon
           return np.random.randint(self.act_num)
        else:
           # Choose the action with highest Q-value at the current state
           action = self.qf(obs).argmax()

        return action.detach().cpu().numpy()

    def train_model(self):
        batch = self.replay_buffer.sample(self.args.batch_size)
        obs1 = batch['obs1']
        obs2 = batch['obs2']
        acts = batch['acts']
        rews = batch['rews']
        done = batch['done']

        if 0: # Check shape of experiences
            print("obs1", obs1.shape)
            print("obs2", obs2.shape)
            print("acts", acts.shape)
            print("rews", rews.shape)
            print("done", done.shape)

        # Prediction Q(s)
        q = self.qf(obs1).gather(1, acts.long()).squeeze(1)
      
        # Target for Q regression
        if self.args.algo == 'dqn':      # DQN
            q_target = self.qf_target(obs2)
        elif self.args.algo == 'ddqn':   # Double DQN
            q2 = self.qf(obs2)
            q_target = self.qf_target(obs2)
            q_target = q_target.gather(1, q2.max(1)[1].unsqueeze(1))
            #print('DDQN')

        q_backup = rews + self.args.gamma*(1-done)*q_target.max(1)[0]
        q_backup.to(self.device)

        if 0: # Check shape of prediction and target
            print("q", q.shape)
            print("q_backup", q_backup.shape)

        # Update perdiction network parameter
        qf_loss = F.mse_loss(q, q_backup.detach())
        #F.huber_loss(q, q_backup.detach()) ###Change to Huber loss
        #F.mse_loss(q, q_backup.detach())
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        # Synchronize target parameters 𝜃‾ as 𝜃 every C steps
        if self.steps % self.args.target_update_step == 0:
            hard_target_update(self.qf, self.qf_target)
      
        # return loss
        return qf_loss.item()

    def run_episode(self, max_step, max_interactions, writer, log_eval, log_state_visitation, q_losses_all_runs, training_eps_seed=None, eval_eps_seed=None,):
        step_number = 0
        total_reward = 0
        #global log_state_visitation, q_losses_all_runs
        
        #AKP_ADDED: Add this to handle the starting state for all algorithm
        if self.eval_mode:
            self.env.seed(eval_eps_seed)
        else:
            self.env.seed(training_eps_seed)
        
        obs = self.env.reset()
        done = False
        ###Added

        # Keep interacting until agent reaches a terminal state.
        while not (done or step_number == max_step or self.steps == max_interactions):
            if self.args.render:
                self.env.render()      

            if self.eval_mode:
                q_value = self.qf(torch.Tensor(obs).to(self.device)).argmax()
                action = q_value.detach().cpu().numpy()
                next_obs, reward, done, _ = self.env.step(action)
                log_state_visitation['eval'][self.run].append(next_obs) #VJN_ADDED & #AKP_MODIFY
            
            else:
                self.steps += 1
                log_state_visitation['train'][self.run].append(obs) #VJN_ADDED & #AKP_MODIFY

                # Collect experience (s, a, r, s') using some policy
                action = self.select_action(torch.Tensor(obs).to(self.device))
                next_obs, reward, done, _ = self.env.step(action)
                ###Added: State Visitation
                log_state_visitation['train'][self.run].append(next_obs)
            
                # Add experience to replay buffer
                self.replay_buffer.add(obs, action, reward, next_obs, done)
                
                # Start training when the number of experience is greater than replay start size and batch size
                if self.steps > self.args.batch_size and self.steps > self.args.replay_start_size: #AKP_MODIFY replay_start_size

                    #AKP_ADDED: train the agent using replay per step and adding qf_losses for each update
                    qf_losses_with_freq = list()
                    for _ in range(self.args.replay_frequency):
                        qf_loss = self.train_model()
                        qf_losses_with_freq.append(qf_loss) #VJN_ADDED
                    self.q_losses[self.steps] = qf_losses_with_freq

                    # log the Q-loss (q_loss averaged over replay frequency for this time step vs time step)
                    if self.args.tensorboard and self.args.load is None:
                        writer.add_scalar('Train/Qloss (averaged over replay frequency for this step vs this step)', np.mean(qf_losses_with_freq), self.steps) ###Change: 
                        q_losses_all_runs[self.run] = self.q_losses
                        

                #AKP_ADDED: Evaluate the agent on a fixed number of interactions, not episodes.
                if self.steps % self.args.eval_per_train == 0:
                    if self.args.console_output == 1:
                        print('###############################################################')
                        print('Offline evaluation episodes start')
                    self.seed_eval_index = 0
                    log_eval = self.eval(writer,log_eval, log_state_visitation, q_losses_all_runs)
                    self.eval_mode = False
                    if self.args.console_output == 1:
                        print('Offline evaluation episodes end')
                        print('###############################################################')
                ###Added: Save Models
                if self.steps % self.args.save_model_per_step == 0:
                    file_path = self.nn_params_path + '/Q_Network'+ '_exp_' + str(self.exp) + '_run_'+ str(self.run) + '_replay_frequency_'+ str(self.args.replay_frequency) + '_steps_' + str(self.steps) + '.pt'
                    save_model(self.qf, file_path)
                    file_path = self.nn_params_path + '/Target_Q_Network'+ '_exp_' + str(self.exp) + '_run_'+ str(self.run) + '_replay_frequency_'+ str(self.args.replay_frequency) + '_steps_' + str(self.steps) + '.pt'
                    save_model(self.qf_target, file_path)

            total_reward += reward
            step_number += 1
            obs = next_obs
      
        # Save logs
        self.logger['LossQ averaged over time steps till now and replay frequency'] = round(np.mean(list(self.q_losses.values())), 5)
        self.logger['Epsilon']= self.args.epsilon
        return step_number, total_reward, log_eval, log_state_visitation, q_losses_all_runs

    #AKP_MODIFY: Evaluate agent after fixed number of interactions, not episodes
    #def eval(agent,args, writer, train_num_steps,train_num_episodes,train_episode_return,train_average_return, start_time):
    def eval(self, writer,log_eval, log_state_visitation, q_losses_all_runs):
        # Perform the evaluation phase -- no learning
        #if (i + 1) % args.eval_per_train == 0:
        #global log_eval
        
        ###Change: to agent.steps
        if self.steps % self.args.eval_per_train == 0: 
            eval_sum_returns = 0.
            eval_num_episodes = 0
            self.eval_mode = True

            for _ in range(self.args.evaluation_episodes):
                # Run one episode
                eval_eps_seed = self.seeds_array_eval[self.seed_eval_index]
                eval_step_length, eval_episode_return, _, _, _ = self.run_episode(self.args.max_step, self.args.max_interactions, writer, log_eval, log_state_visitation, q_losses_all_runs, eval_eps_seed=eval_eps_seed) ###Change: 
                self.seed_eval_index += 1

                eval_sum_returns += eval_episode_return
                eval_num_episodes += 1
                
                if self.args.console_output == 1:
                    print('---------------------------------------')
                    #print('Interactions in this offline eval episode:', eval_step_length)
                    print('Episodes:', eval_num_episodes)
                    print('EpisodeReturn:', round(eval_episode_return, 2))
                    print('---------------------------------------')

            eval_average_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0

            # Log experiment result for evaluation episodes
            if self.args.tensorboard and self.args.load is None:
                writer.add_scalar('Eval/AverageReturns', eval_average_return, self.steps) ###Change: 
                writer.flush()
                #writer.add_scalar('Eval/EpisodeReturns', eval_episode_return, self.steps) ###Change:
                df = pd.DataFrame({'Metric': 'AverageReturns', 'Run': self.run, 'Step': self.steps, 'Undiscount_Return': eval_average_return}, index=[0])
                log_eval = pd.concat([log_eval.loc[:], df]).reset_index(drop=True)
                #log_eval = log_eval.append({'Metric': 'EpisodeReturns', 'Run': self.run, 'Step': self.steps, 'Undiscount_Return': eval_episode_return}, ignore_index = True)
                
                

            if self.args.phase == 'train' and self.args.console_output == 1:
                print('---------------------------------------')
                print('EvalEpisodes:', eval_num_episodes)
                #print('EvalEpisodeReturn:', round(eval_episode_return, 2))
                print('EvalAverageReturn:', round(eval_average_return, 2))
                print('---------------------------------------')

                # Save the trained model
                if eval_average_return >= self.args.threshold_return:
                    ###Changed:
                    file_path1 = 'Eval_' + str(round(eval_average_return, 2)) + 'Q_Network' + '_exp_' + str(self.args.exp) + '_run_'+ str(self.run) +'_replay_frequency_'+ str(self.args.replay_frequency) + '_steps_' + str(self.steps) + '.pt'
                    file_path2 = 'Eval_' + str(round(eval_average_return, 2)) + 'Target_Q_Network' + '_exp_' + str(self.args.exp) + '_run_'+ str(self.run) +'_replay_frequency_'+ str(self.args.replay_frequency) + '_steps_' + str(self.steps) + '.pt'
                    if self.args.algo == 'dqn' or self.args.algo == 'ddqn':
                        #torch.save(self.qf.cpu(), ckpt_path)
                        save_model(self.qf, file_path1)
                        save_model(self.qf_target, file_path2)
                    else:
                        #torch.save(self.policy.cpu(), ckpt_path)
                        save_model(self.policy, file_path1)
        return log_eval

        