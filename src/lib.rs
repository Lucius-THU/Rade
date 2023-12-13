use std::sync::atomic::{AtomicBool, Ordering};

pub mod cpu;
pub mod device;
pub mod functional;
mod ndarray;
pub mod nn;
mod operation;
pub mod optim;
pub mod tensor;
mod type_trait;

static LAZY_MODE: AtomicBool = AtomicBool::new(false);

pub fn is_lazy() -> bool {
    LAZY_MODE.load(Ordering::Relaxed)
}

pub fn set_lazy_mode(flag: bool) {
    LAZY_MODE.store(flag, Ordering::Relaxed);
}
