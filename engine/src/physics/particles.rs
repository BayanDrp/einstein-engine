use crate::geometry::christoffel::ChristoffelField;
use crate::numerics::interpolation::InterpolationError;
use crate::physics::geodesics::{geodesic_stages_step_rk4, GeodesicState, StageConnections};

/// One committed state of a particle, stamped with its coordinate time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleSample<const D: usize> {
    pub t: f64,
    pub pos: [f64; D],
    pub vel: [f64; D],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParticleStatus {
    Active,
    StoppedAtBoundary,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Particle<const D: usize> {
    pub id: u64,
    pub time: f64,
    pub state: GeodesicState<D>,
    /// One [`ParticleSample`] per committed step; grows without bound over
    /// long runs.
    pub history: Vec<ParticleSample<D>>,
    pub status: ParticleStatus,
}

#[derive(Debug)]
pub enum ParticleError {
    InvalidTimeStep,
    NonFiniteTime,
    NonFiniteState,
    Interpolation(InterpolationError),
    /// A synchronized step was called with a coupling time `t` that an
    /// active particle does not share: the particle sits at its own time
    /// stamp while the ADM data is stamped `coupling_time`. Nothing is
    /// stepped when this fires.
    TimeMismatch {
        particle_id: u64,
        coupling_time: f64,
    },
}

impl From<InterpolationError> for ParticleError {
    fn from(error: InterpolationError) -> Self {
        Self::Interpolation(error)
    }
}

fn valid_state<const D: usize>(state: &GeodesicState<D>) -> bool {
    state
        .pos
        .iter()
        .chain(state.vel.iter())
        .all(|value| value.is_finite())
}

fn validate_dt(dt: f64) -> Result<(), ParticleError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(ParticleError::InvalidTimeStep);
    }

    Ok(())
}

impl<const D: usize> Particle<D> {
    fn new(id: u64, time: f64, state: GeodesicState<D>) -> Self {
        let initial_sample = ParticleSample {
            t: time,
            pos: state.pos,
            vel: state.vel,
        };

        Self {
            id,
            time,
            state,
            history: vec![initial_sample],
            status: ParticleStatus::Active,
        }
    }

    fn commit_advance(&mut self, next_time: f64, new_state: GeodesicState<D>) {
        self.time = next_time;
        self.state = new_state;

        self.history.push(ParticleSample {
            t: self.time,
            pos: self.state.pos,
            vel: self.state.vel,
        });
    }
}

/// Outcome computed for one particle before anything is mutated.
enum PendingStep<const D: usize> {
    Advance(f64, GeodesicState<D>),
    StopAtBoundary,
    Skip,
}

/// A collection of test particles evolved on a static connection.
#[derive(Debug, Clone, PartialEq)]
pub struct Particles<const D: usize> {
    particles: Vec<Particle<D>>,
    next_id: u64,
}

impl<const D: usize> Default for Particles<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> Particles<D> {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            next_id: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.particles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Particle<D>> {
        self.particles.iter()
    }

    /// Look a particle up by the id returned from [`Particles::spawn`].
    pub fn get(&self, id: u64) -> Option<&Particle<D>> {
        self.particles.iter().find(|particle| particle.id == id)
    }

    /// Add a particle at `time` and return its id.
    ///
    /// Rejects non-finite time or state up front; the returned particle is
    /// always ready to step.
    pub fn spawn(&mut self, time: f64, state: GeodesicState<D>) -> Result<u64, ParticleError> {
        if !time.is_finite() {
            return Err(ParticleError::NonFiniteTime);
        }

        if !valid_state(&state) {
            return Err(ParticleError::NonFiniteState);
        }

        let id = self.next_id;
        self.next_id += 1;

        self.particles.push(Particle::new(id, time, state));

        Ok(id)
    }

    /// Advance every active particle by one RK4 step of size `dt`.
    ///
    /// A particle whose step leaves the interpolation domain is marked
    /// [`ParticleStatus::StoppedAtBoundary`] and keeps its last valid state
    /// and timestamp; the remaining particles still advance and the call
    /// succeeds. Any other failure (non-finite time or state) returns `Err`
    /// with every particle left untouched: candidates are computed for all
    /// active particles before any particle is mutated.
    pub fn step_all<const N: usize>(
        &mut self,
        dt: f64,
        gamma: &ChristoffelField<N, D>,
    ) -> Result<(), ParticleError> {
        validate_dt(dt)?;

        let frozen = [gamma, gamma, gamma, gamma];

        self.step_all_via(dt, |state| geodesic_stages_step_rk4(state, dt, frozen))
    }

    /// Advance every active particle by one synchronized step: `connections`
    /// was built from one ADM RK4 step whose starting time is `t`, so every
    /// active particle must currently sit at that same time before it is
    /// allowed to step by `connections.dt`.
    ///
    /// The clock check is deliberately lenient (`1e-9 * (1 + |t|)`, ample
    /// room for floating-point noise) and refuses the whole batch with
    /// [`ParticleError::TimeMismatch`] before anything is touched; the ADM
    /// data would be silently resampled at the wrong stage times
    /// otherwise, which is a caller bug, not a physics state. Stopped
    /// particles carry their last valid timestamp and are exempt: they do
    /// not step and nobody re-synchronizes them.
    ///
    /// Same transactional semantics as [`Particles::step_all`]: a particle
    /// leaving the interpolation domain stops at its last valid state
    /// while the rest advance; anything else fails with everything
    /// untouched.
    pub fn step_all_with_connections<const N: usize>(
        &mut self,
        t: f64,
        connections: &StageConnections<N, D>,
    ) -> Result<(), ParticleError> {
        if !t.is_finite() {
            return Err(ParticleError::NonFiniteTime);
        }

        validate_dt(connections.dt)?;

        let tolerance = 1e-9 * (1.0 + t.abs());
        for particle in &self.particles {
            if particle.status != ParticleStatus::Active {
                continue;
            }

            if !particle.time.is_finite() {
                return Err(ParticleError::NonFiniteTime);
            }

            if (particle.time - t).abs() > tolerance {
                return Err(ParticleError::TimeMismatch {
                    particle_id: particle.id,
                    coupling_time: t,
                });
            }
        }

        let bundles = [
            &connections.connections[0],
            &connections.connections[1],
            &connections.connections[2],
            &connections.connections[3],
        ];
        let dt = connections.dt;

        self.step_all_via(dt, |state| geodesic_stages_step_rk4(state, dt, bundles))
    }

    /// Compute-then-commit core shared by both entry points: the RK4
    /// recipe per particle is supplied by `step_one` (frozen vs
    /// stage-synchronized bundles); validation, the
    /// stop-at-domain-edge policy and the all-or-nothing commit are
    /// identical for both.
    fn step_all_via(
        &mut self,
        dt: f64,
        step_one: impl Fn(&GeodesicState<D>) -> Result<GeodesicState<D>, InterpolationError>,
    ) -> Result<(), ParticleError> {
        let mut pending: Vec<PendingStep<D>> = Vec::with_capacity(self.particles.len());

        for particle in &self.particles {
            if particle.status != ParticleStatus::Active {
                pending.push(PendingStep::Skip);
                continue;
            }

            let next_time = particle.time + dt;

            if !next_time.is_finite() {
                return Err(ParticleError::NonFiniteTime);
            }

            match step_one(&particle.state) {
                Ok(next_state) => {
                    if !valid_state(&next_state) {
                        return Err(ParticleError::NonFiniteState);
                    }

                    pending.push(PendingStep::Advance(next_time, next_state));
                }
                // Leaving the interpolation domain stops this particle at
                // its last valid state. `InterpolationError` currently has
                // only this variant, so a future one fails to compile here
                // on purpose: whether it should stop or abort is a policy
                // decision, not an accident.
                Err(InterpolationError::OutsideDomain) => {
                    pending.push(PendingStep::StopAtBoundary);
                }
            }
        }

        for (particle, outcome) in self.particles.iter_mut().zip(pending) {
            match outcome {
                PendingStep::Advance(next_time, next_state) => {
                    particle.commit_advance(next_time, next_state);
                }
                PendingStep::StopAtBoundary => {
                    particle.status = ParticleStatus::StoppedAtBoundary;
                }
                PendingStep::Skip => {}
            }
        }

        Ok(())
    }
}
