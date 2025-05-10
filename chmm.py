import numpy as np
from typing import Literal, Optional

# available initialization for transition matrix
INI_MODE = Literal['dirichlet', 'normal', 'uniform']


def softmax(x: np.ndarray, temp=1.) -> np.ndarray:
    """Computes softmax values for a vector `x` with a given temperature."""
    temp = np.clip(temp, 1e-5, 1e+3)
    e_x = np.exp((x - np.max(x, axis=-1)) / temp)
    return e_x / e_x.sum(axis=-1)


class CHMM:
    def __init__(
            self,
            n_columns: int,
            cells_per_column: int,
            lr: float = 0.1,
            batch_size: int = 1,
            initialization: INI_MODE = 'uniform',
            sigma: float = 1.0,
            alpha: float = 1.0,
            seed: Optional[int] = None
    ):
        """
        Custom realization of CHMM method from paper https://arxiv.org/abs/1905.00507.
        This is just a template, you can change this class as needed.

        n_columns:
        cells_per_columns: number of hidden state copies for an observation state, we also call them `columns`
        lr: learning rate for matrix updates
        batch_size: sequence size for learning
        initialization: transition matrix initialization
        sigma: parameter of normal distribution
        alpha: parameter of alpha distribution
        seed: seed for reproducibility, None means no reproducibility
        """

        self.n_columns = n_columns
        self.cells_per_column = cells_per_column
        self.n_states = cells_per_column * n_columns
        self.lr = lr
        self.batch_size = batch_size
        self.initialization = initialization
        self.is_first = True
        self.seed = seed

        self._rng = np.random.default_rng(self.seed)

        if self.initialization == 'dirichlet':
            self.transition_probs = self._rng.dirichlet(
                alpha=[alpha]*self.n_states,
                size=self.n_states
            )
            self.state_prior = self._rng.dirichlet(alpha=[alpha]*self.n_states)
        elif self.initialization == 'normal':
            self.log_transition_factors = self._rng.normal(
                scale=sigma,
                size=(self.n_states, self.n_states)
            )
            self.log_state_prior = self._rng.normal(scale=sigma, size=self.n_states)
        elif self.initialization == 'uniform':
            self.log_transition_factors = np.zeros((self.n_states, self.n_states))
            self.log_state_prior = np.zeros(self.n_states)

        if self.initialization != 'dirichlet':
            self.transition_probs = np.vstack(
                [softmax(x) for x in self.log_transition_factors]
            )

            self.state_prior = softmax(self.log_state_prior)
        else:
            self.log_transition_factors = np.log(self.transition_probs)
            self.log_state_prior = np.log(self.state_prior)


        self.batch_observations = []
        self.seq_observations = []
        self.A = self.transition_probs
        self.seq_ksi_batch = np.zeros((self.n_states, self.n_states))
        self.current_alpha = np.zeros(self.cells_per_column)

        self.batch_alphas = []
        self.seq_alphas = []
        
        self.batch_betas = []
        self.state_posterior = self.state_prior
                
    def get_clone_indices(self, x: int) -> np.ndarray:
        return np.arange(x*self.cells_per_column, (x+1)*self.cells_per_column)        
    
    def observe(self, observation_state: int, learn: bool) -> None:
        
        if learn:
            self.process_batch()
            
        self.batch_observations.append(observation_state)
        self.seq_observations.append(observation_state)
        
        self.is_first = False
                
        curr_x = self.get_clone_indices(observation_state)
        if len(self.seq_alphas) == 0:
            alpha_n = self.state_prior[curr_x]
        else:
            prev_x = self.get_clone_indices(self.batch_observations[-2])
            alpha_n = self.seq_alphas[-1] @ self.transition_probs[np.ix_(prev_x, curr_x)]

        self.seq_alphas.append(alpha_n) 
        self.batch_alphas.append(alpha_n) 
        
        self.state_posterior[curr_x] = alpha_n
        self.state_posterior /= sum(self.state_posterior)
        
    
    def backward(self) -> None:
        beta = np.ones((len(self.seq_observations), self.cells_per_column))
        beta[len(self.seq_observations) - 1] = 1.0
        
        for n in range(len(self.seq_observations) - 2, -1, -1):
            curr_x = self.get_clone_indices(self.seq_observations[n])
            next_x = self.get_clone_indices(self.seq_observations[n+1])
            beta[n] = self.transition_probs[np.ix_(curr_x, next_x)] @ beta[n+1]
        
        for row in range(len(self.seq_observations)):
            self.batch_betas.append(beta[row])
                    
    def process_batch(self) -> None:
            
        gamma = ((self.batch_alphas[0] * self.batch_betas[0]) / (self.batch_alphas[0] @ self.batch_betas[0]))
        self.state_prior[self.get_clone_indices(self.batch_observations[0])] = gamma
        self.state_prior /= sum(self.state_prior)
                
        
        for n in range(len(self.batch_observations) - 1):
            rows = self.get_clone_indices(self.batch_observations[n])
            cols = self.get_clone_indices(self.batch_observations[n + 1])
            
            T_sub = self.transition_probs[np.ix_(rows, cols)]
            self.seq_ksi_batch[np.ix_(rows, cols)] += (self.batch_alphas[n][:, None] * T_sub * self.batch_betas[n + 1][None, :]) / (self.batch_alphas[n] @ T_sub @ self.batch_betas[n + 1])
            self.A[np.ix_(rows, cols)] = self.lr * self.A[np.ix_(rows, cols)] + (1 - self.lr) * self.seq_ksi_batch[np.ix_(rows, cols)]
            total_sums = self.A[np.ix_(rows, cols)].sum(axis=1)
            
            self.transition_probs[np.ix_(rows, cols)] = self.A[np.ix_(rows, cols)] / total_sums[:, None]                
                
        self.transition_probs = self.transition_probs / self.transition_probs.sum(axis=1, keepdims=True)

        self.seq_ksi_batch = np.zeros((self.n_states, self.n_states))
        self.batch_observations = []
        self.seq_alphas = []
        self.batch_alphas = []
        self.batch_betas = []
        
    def predict_observation_states(self, observation_state: int) -> np.ndarray:
        
        curr_x = self.get_clone_indices(observation_state)
        
        observation_mass = np.zeros((self.n_columns, self.cells_per_column))
        if self.is_first:
            self.current_alpha = self.state_prior[curr_x]
        else:
            self.current_alpha = self.seq_alphas[-1]

        
        for l in range(self.n_columns):
            cols = self.get_clone_indices(l)
            observation_mass[l] = (self.current_alpha @ self.transition_probs[np.ix_(curr_x, cols)])
        
        vect_probs = observation_mass.flatten()
        vect_probs /= sum(vect_probs)
        
        observation_probs = [sum(vect_probs[i : i + self.cells_per_column]) for i in range(0, len(vect_probs), self.cells_per_column)]
        
        return observation_probs / sum(observation_probs)
    

    def reset(self) -> None:        
        self.seq_alphas = []
        self.seq_observations = []
        self.state_posterior = self.state_prior
        self.is_first = True
        
        
        
