//#![deny(unsafe_code, missing_docs, unused_imports)]
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

//! Implementation of a Kalman filter with all statistical bells and whistles exposed.

use nalgebra::{
    VectorN,
    MatrixN,
    Scalar,
    Dim,
    DimName,
    DefaultAllocator,
    allocator::Allocator,
    Real,
};

#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

///
// note: serde broken -- #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "compact", repr(packed))]
#[derive(Clone, Debug)]
pub struct Kalman<T: Scalar, D: Dim + DimName>
    where DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>
{
    /// x_k: current position of the system.
    state: VectorN<T, D>,

    /// P_k: uncertainty about x_k.
    uncertainty: MatrixN<T, D>,

    /// Q_k: noise added to the uncertainty
    environment_noise: MatrixN<T, D>,

    /// H_k: maps real space into observation space
    obs_map: MatrixN<T, D>,

    #[cfg(not(feature = "compact"))]
    /// H_k^T: transpose of the observation map. Not stored in compact mode.
    obs_transpose: MatrixN<T, D>,

    /// R_k: noise added to the observation
    obs_noise: MatrixN<T, D>,
}

impl <T: Scalar + Real, D: Dim + DimName> Kalman<T, D>
    where DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>
{
    pub fn new(initial_state:           impl Into<VectorN<T, D>>,
               initial_uncertainty:     impl Into<MatrixN<T, D>>,
               obs_map:                 impl Into<MatrixN<T, D>>,
               obs_noise:               impl Into<MatrixN<T, D>>,
               noise:                   impl Into<MatrixN<T, D>>) -> Self
    {
        #[cfg(feature = "compact")] {
            Kalman {
                state: initial_state.into(),
                uncertainty: initial_uncertainty.into(),
                environment_noise: noise.into(),
                obs_map: obs_map.into(),
                obs_noise: obs_noise.into(),
            }
        }

        #[cfg(not(feature = "compact"))] {
            let obs_map = obs_map.into();
            let obs_transpose = obs_map.transpose();

            Kalman {
                state: initial_state.into(),
                uncertainty: initial_uncertainty.into(),
                environment_noise: noise.into(),
                obs_map,
                obs_transpose,
                obs_noise: obs_noise.into(),
            }
        }
    }

    /// Predict the next state.
    /// `control` is the control signal, which is added directly to the final estimate. This is used to
    /// represent information we know about external influences on the system that are not accounted
    /// for by the estimator matrix.
    pub fn predict(&mut self, transition: impl Into<MatrixN<T, D>>, control: impl Into<VectorN<T, D>>) {
        let transition = transition.into();
        let control = control.into();

        let new_state = &transition * &self.state + control;
        let new_uncertainty = &transition * &self.uncertainty * &transition.transpose() + &self.environment_noise;

        self.state = new_state;
        self.uncertainty = new_uncertainty;
    }

    pub fn update(&mut self, observation: impl Into<VectorN<T, D>>) {
        #[cfg(feature = "compact")]
        let obs_transpose = &self.obs_map.transpose();

        #[cfg(not(feature = "compact"))]
        let obs_transpose = &self.obs_transpose;

        let kalman_gain = &self.uncertainty * obs_transpose * (&self.obs_map * &self.uncertainty * obs_transpose + &self.obs_noise).try_inverse().unwrap();

        self.state = &self.state + &kalman_gain * (observation.into() - &self.obs_map * &self.state);
        self.uncertainty = &self.uncertainty - kalman_gain * &self.obs_map * &self.uncertainty;
    }
}
