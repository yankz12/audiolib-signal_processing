import numpy as np

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass

import pdb

@dataclass(kw_only=True)
class StateSpaceModelling(ABC):
    """
    Calculates the dynamics of an Nth ordder system as a first order
    differential equation in an N-sized-vector (= state):
    
    dq(t)/dt = A*q(t) + B*u(t),
        q(t) being the state vector and
        u(t) being the input vector

    It is intented to always have the output as dictionary in self.output_dict
    The pure output matrix is ommited.
    Setting the observation order resets the output matrix to zeros!

    Parameters
    ----------
    A : np.nparray
        A-Matrix of the system under study
    B : np.ndarray
    input_sig : input signal
    input_time : time vector of input signal (sampling freq. is derived from
        this vector)
    obs_order : list of strings
        CAUTION: Setting this value after initialization resets output to zeros!
        Contains the order of observations, e.g. ['i', 'x', 'v'] for observing
        current, displacement and velocity. Is directly linked to matrices
        A & B, since the observation order changes the shape of those matrices
    """
    A: np.ndarray
    B: np.ndarray
    input_sig : np.ndarray # Input signal fitting to input_time
    input_time : np.ndarray # Time vector of input, fs is defined by setting input_time
    obs_order : list[str] # depends on structure of A and B

    # def __post_init__(self, obs_order: list, ):
    #     print('Reach post')
    #     self.obs_order = obs_order

    @abstractmethod
    def run_over_input(self):
        pass

    @abstractmethod
    def run_one_sample(self):
        pass

    def _zero_output_matrix(self):
        self._output_matrix = np.zeros((len(self.input_time)+1,self._num_obs))

    def _validate_input_sig(self):
        if self.input_sig is None:
            raise ValueError(
                "No input vector given. Consider using run_one_sample()."
            )
        if self.input_time is None:
            raise ValueError(
                "No input time vector given. Consider using run_one_sample()."
            )
        
    def is_nonlinear(self):
        # Check if an entry of A-matrix is a function. If so, it is non-linear
        for entry in np.nditer(self.A):
            if callable(entry):
                return True
        return False
    
    def get_nonlinear_funcs(self):
        if not self.is_nonlinear():
            raise ValueError(
                'System is linear: no non-linear functions to return.'
            )
        nonlin_entries = np.array([])
        for entry in np.nditer(self.A):
            if callable(entry):
                nonlin_entries.append(entry)
        return nonlin_entries
    
    def tmp_nonlin_result_matrix(self):
        nonlin_funcs = self.get_nonlinear_funcs()
        for func in nonlin_funcs:
            tmp_A = self.A
            tmp_A[np.where(self.A==func)] = func(self.output_dict)
        return tmp_A

    @property
    def output_dict(self):
        return OrderedDict(
            zip(self.obs_order, self._output_matrix.T)
        ) # Make output read-only

    @property
    def input_time(self):
        return self._input_time
    
    @input_time.setter
    def input_time(self, time, ):
        self._input_time = time
        self.Ts = self.input_time[1] - self.input_time[0]
        self.fs = int(np.round(1/self.Ts))
    
    @property
    def obs_order(self):
        return self._obs_order
    
    @obs_order.setter
    def obs_order(self, order, ):
        self._obs_order = order
        self._num_obs = len(order)
        self._ident = np.identity(self._num_obs)
        self._zero_output_matrix()


@dataclass(kw_only=True)
class EulerBackward(StateSpaceModelling):
    """
    Make sure to delay your input signal by one sample, since Euler Backward
    will skip input[0] and start directly with input[1] due to its 
    discretization nature ("previous output" doesn't exist when first input
    is "current input", see formula below)!

    X[n+1] = (I - A*Ts)^{-1} * X[n] + (I - A*Ts)^{-1} * B * Ts * U[n+1]
    with
        X[n+1] = current output
        X[n] = previous output
        U[n+1] = current input

    Parameters
    ----------
    A : np.nparray
        A-Matrix of the system under study
    B : np.ndarray
    input_sig : input signal
    input_time : time vector of input signal (sampling freq. is derived from
        this vector)
    obs_order : list of strings
        CAUTION: Setting this value after initialization resets output to zeros!
        Contains the order of observations, e.g. ['i', 'x', 'v'] for observing
        current, displacement and velocity. Is directly linked to matrices
        A & B, since the observation order changes the shape of those matrices
    """

    def __post_init__(self):
        self._A_new = np.linalg.inv((self._ident - self.A*self.Ts)) # new A matrix
        self._B_new = self._A_new @ (self.B * self.Ts) # new B matrix

    def run_over_input(self):
        self._validate_input_sig()
        # range(1, ...-1) to give space for future input signal
        for idx in range(1, len(self._output_matrix)-1):
            y_n = self.run_one_sample(idx)
            self._output_matrix[idx] = y_n

    def run_one_sample(self, idx, ):
        y_n = (
            self._A_new @ self._output_matrix[idx-1] +
            self._B_new * self.input_sig[idx]
        )
        return y_n


@dataclass(kw_only=True)
class Bilinear(StateSpaceModelling):
    """
    X[n+1] = (I - A*Ts/2)^{-1} * (I + A*Ts/2) * X[n] +
             (I - A*Ts/2)^{-1} * B * Ts/2 * (U[n+1] + U[n])
    with
    A_new = (I - A*Ts/2)^{-1} * (I + A*Ts/2)
    and
    B_new = (I - A*Ts/2)^{-1} * B*Ts/2
    we get
    X[n+1] = A_new * X[n] + B_new * (U[n+1] + U[n]) / 2
    with
        X[n+1] = current output
        X[n] = previous output
        U[n+1] = current input
        U[n] = previous input

    Parameters
    ----------
    A : np.nparray
        A-Matrix of the system under study
    B : np.ndarray
    input_sig : input signal
    input_time : time vector of input signal (sampling freq. is derived from
        this vector)
    obs_order : list of strings
        CAUTION: Setting this value after initialization resets output to zeros!
        Contains the order of observations, e.g. ['i', 'x', 'v'] for observing
        current, displacement and velocity. Is directly linked to matrices
        A & B, since the observation order changes the shape of those matrices
    """

    def __post_init__(self):
        """
        Save inversion results in temporary variable to not have to
        invert twice: inversion is computationally intensive!
        """
        inv_neg = np.linalg.inv(self._ident - self.A*self.Ts/2)
        self._A_new = inv_neg @ (self._ident + self.A*self.Ts/2)
        self._B_new = inv_neg @ self.B*self.Ts

    def run_over_input(self):
        self._validate_input_sig()
        for idx in range(1, len(self._output_matrix)-1):
            y_n = self.run_one_sample(idx)
            self._output_matrix[idx] = y_n

    def run_one_sample(self, idx, ):
        y_n = (
            self._A_new @ self._output_matrix[idx - 1] +
            self._B_new * (self.input_sig[idx] + self.input_sig[idx-1])/2
        )
        return y_n


@dataclass(kw_only=True)
class EulerForward(StateSpaceModelling):
    """
    X[n+1] = (A*Ts + I) * X[n] + B * Ts * U[n]
    with
        A_new = (A * Ts + I)
        B_new = B * Ts
    X[n+1] = A_new * X[n] + B_new * U[n]
    with
        X[n+1] = current output
        X[n] = previous output
        U[n] = previous input

    Parameters
    ----------
    A : np.nparray
        A-Matrix of the system under study
    B : np.ndarray
    input_sig : input signal
    input_time : time vector of input signal (sampling freq. is derived from
        this vector)
    obs_order : list of strings
        CAUTION: Setting this value after initialization resets output to zeros!
        Contains the order of observations, e.g. ['i', 'x', 'v'] for observing
        current, displacement and velocity. Is directly linked to matrices
        A & B, since the observation order changes the shape of those matrices
    """
    def __post_init__(self):
        # if self.is_nonlinear():
        self._A_new = self.A*self.Ts + self._ident # New A matrix
        self._B_new = self.B*self.Ts # New B matrix

    def run_over_input(self, ):
        self._validate_input_sig() # TODO: Move to super class setter of input?
        for idx in range(1, len(self._output_matrix)):
            y_n, _ = self.run_one_sample(idx)
            self._output_matrix[idx] = y_n

    def run_one_sample(self, idx, ):
        f_n1 = self.A @ self._output_matrix[idx-1] + self.B * self.input_sig[idx-1]
        y_n = self._output_matrix[idx-1] + self.Ts*f_n1
        return y_n, f_n1


@dataclass(kw_only=True)
class AdamBashforth(StateSpaceModelling):
    """
    Step 1: X[n+1] = X[n] * (I + A*Ts) + B*Ts*U[n] (= EulerForward)
    Step 2: x[n+2] = X[n+1] * (I + 3/2*A*Ts) + Ts{ B*U[n+1] - .5*(A*X[n] + B*U[n]) }
    Step 3: X[n+3] = X[n+2] * (I + 23/12*A*Ts) + Ts{
        B*U[n+2] - 16/12*(A*X[n+1] + B*U[n+1]) + 5/12*(A*X[n] + B*U[n])
    }

    with
    
    X[n+3] = X[n]
    X[n+2] = X[n-1]
    X[n+1] = X[n-2]
    X[n]   = X[n-3]
    
    X[n-2] = X[n-3] * (I + A*Ts) + B*Ts*U[n-3] (= EulerForward)
    x[n-1] = X[n-2] * (I + 3/2*A*Ts) + Ts{ B*U[n-2] - .5*(A*X[n-3] + B*U[n-3]) }
    X[n]   = X[n-1] * (I + 23/12*A*Ts) + Ts{
        B*U[n-1] - 16/12*(A*X[n-2] + B*U[n-2]) + 5/12*(A*X[n-3] + B*U[n-3])
    }

    Parameters
    ----------
    A : np.nparray
        A-Matrix of the system under study
    B : np.ndarray
    input_sig : input signal
    input_time : time vector of input signal (sampling freq. is derived from
        this vector)
    obs_order : list of strings
        CAUTION: Setting this value after initialization resets output to zeros!
        Contains the order of observations, e.g. ['i', 'x', 'v'] for observing
        current, displacement and velocity. Is directly linked to matrices
        A & B, since the observation order changes the shape of those matrices
    """
    order : int = 3

    def __post_init__(self,):
        # Euler Forward for initialisation of first sample
        self._euler_forward = EulerForward(
            A=self.A,
            B=self.B,
            input_sig=self.input_sig,
            input_time=self.input_time,
            obs_order=self.obs_order,
        )

    def _validate_order(self, ord, ):
        if ord > 3 or ord < 2:
            raise ValueError(
                f"Order {ord} out of range: has to be 2 or 3."
            )

    def _run_order_2(self, idx, ):
        # f_n2 = self.A @ self._output_matrix[idx-2] + self.B * self.input_sig[idx-2]
        f_n2 = self.A @ self._output_matrix[idx-2] + self.B * self.input_sig[idx-2]
        y_n1, f_n2 = self._euler_forward.run_one_sample(idx-1)
        f_n1 = self.A @ self._output_matrix[idx-1] + self.B * self.input_sig[idx-1]
        
        _ = self._output_matrix[idx-2] + self.Ts*f_n2
        y_n  = self._output_matrix[idx-1] + self.Ts/2*(3*f_n1 - f_n2)
        return [y_n1, y_n], [f_n2, f_n1]

    def _run_order_3(self, idx, ):
        """
        Runs 3rd order Adam-Bashforth over input signal, uses index as iterator

        Parameters
        ----------
        idx : int
            index of current position in output-vector (X[idx]),
            idx is the time at which the output/derivative is to be calculated

        Returns
        -------
        outputs : matrix, np.ndarray
            vector of outputs
            [y_{n-2}, y_{n-1}, y_{n}]
        states : np.ndarray
            vector of states
            [f(t_{n-3}, y_{n-3}), f(t_{n-2}, y_{n-2})), f(t_{n-1}, y_{n-1})]

        U[n-2, n-1, n] = [input_0, input_1, input_2, ]
        Y[n-3, n-2, n-1, n] = [output_0, output_1, output_2, output_3, ],
        """
        # sample @ idx-1 because 2nd order initializes states before 3rd order:
        idx_order_2 = idx - 1
        [y_n2, y_n1], [f_n3, f_n2] = self._run_order_2(
            idx = idx_order_2,
        )
        f_n1 = self.A @ self._output_matrix[idx-1] + self.B * self.input_sig[idx-1]
        y_n = self._output_matrix[idx-1] + self.Ts/12*(23*f_n1 - 16*f_n2 + 5*f_n3)
        outputs = [y_n2, y_n1, y_n]
        states = [f_n3, f_n2, f_n1]
        return outputs, states 

    def run_one_sample(self, ):
            # Is overridden by setter of self.order property
            raise ValueError(
                "Need to define order of Adam-Bashforth (2 or 3) before running."
            )

    def run_over_input(self):
        for idx in range(len(self._output_matrix)):
            if idx == 0:
                tmp_idx = self._order + 1
                output, _ = self.run_one_sample(
                    idx = tmp_idx,
                )
                self._output_matrix[:self.order] = output
            elif idx < self._order + 1:
                # Skip steps that have already been handled by "if idx == 0"
                continue 
            else:
                # Write matrix y_n @ idx after initialization to output matrix
                [_, _, y_n], _ = self.run_one_sample(idx)
                self._output_matrix[idx] = y_n
            
            # Need to overwrite euler forward output matrix for correct calcs:
            self._euler_forward._output_matrix = self._output_matrix

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, ord, ):
        self._validate_order(ord)
        self._order = ord
        if ord == 2:
            self.run_one_sample = self._run_order_2
        elif ord == 3:
            self.run_one_sample = self._run_order_3


@dataclass(kw_only=True)
class Heun(StateSpaceModelling):
    """
    Heuns method is a predictor-corrector method to calculate linear and
    non-linear ODEs. It estimates the derivative using Euler-Forward 
    (predictor) and improves the accuracy using Bilinear (corrector).

    ŷ[n+1] = y[n] + Ts*f(t[n], y[n]) (Euler Forward)
    y[n+1] = y[n] + 1/2 * Ts * (f(t[n+1], ŷ[n+1]) + f(t[n], y[n])) (Bilinear)

    which is the same as

    ŷ[n] = y[n-1] + Ts*f(t[n-1], y[n-1]) (Euler Forward)
    y[n] = y[n-1] + 1/2 * Ts * (f(t[n], ŷ[n]) + f(t[n-1], y[n-1])) (Bilinear)
    """

    def __post_init__(self):
        self._euler_forward = EulerForward(
            A=self.A,
            B=self.B,
            input_sig=self.input_sig,
            input_time=self.input_time,
            obs_order=self.obs_order,
        ) # Initialize predictor

    def run_over_input(self):
        self._validate_input_sig() # TODO: Move to super class setter of input?
        for idx in range(1, len(self._output_matrix)-1):
            y_n = self.run_one_sample(idx)
            self._output_matrix[idx] = y_n
            # Need to overwrite euler forward output matrix for correct calcs:
            self._euler_forward._output_matrix = self._output_matrix
    
    def run_one_sample(self, idx, ):
        # "f_n1" means f(t,y) at t-1, "f_n" means f(t,y) at t 
        
        # Predictor calculation
        if self.is_nonlinear():
            tmp_A_pred = self.tmp_nonlin_result_matrix(
                self.output_dict[idx-1]
            )
            self._euler_forward.A = tmp_A_pred
        ŷ_n, f_n1 = self._euler_forward.run_one_sample(idx)

        # Corrector Calculation
        tmp_A_corr = self.tmp_nonlin_result_matrix(
                ŷ_n
            ) if self.is_nonlinear() else self.A
        f_n = tmp_A_corr @ ŷ_n + self.B*self.input_sig[idx]
        y_n = self._output_matrix[idx-1] + .5*self.Ts*(f_n + f_n1) 
        return y_n
    