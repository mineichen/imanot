use std::{pin::Pin, task::Context};

use crate::LocalBoxFuture;

pub enum AsyncRefTask<T> {
    Pending(LocalBoxFuture<'static, T>),
    Ready(T),
}

impl<T> AsyncRefTask<T> {
    pub fn new(b: LocalBoxFuture<'static, T>) -> Self {
        Self::Pending(b)
    }

    pub fn new_ready(r: T) -> Self {
        Self::Ready(r)
    }

    pub fn data(&mut self) -> Option<&mut T> {
        match self {
            AsyncRefTask::Pending(x) => {
                let waker = std::task::Waker::noop();
                let mut cx = Context::from_waker(waker);
                match Pin::new(x).poll(&mut cx) {
                    std::task::Poll::Ready(r) => {
                        *self = Self::Ready(r);
                        let Self::Ready(x) = self else {
                            panic!("Should never be called")
                        };
                        Some(x)
                    }
                    std::task::Poll::Pending => None,
                }
            }
            AsyncRefTask::Ready(x) => Some(x),
        }
    }
}
pub struct AsyncTask<T>(LocalBoxFuture<'static, T>);

impl<T> AsyncTask<T> {
    pub fn new(b: LocalBoxFuture<'static, T>) -> Self {
        Self(b)
    }

    pub fn data(&mut self) -> Option<T> {
        let waker = std::task::Waker::noop();
        let mut cx = Context::from_waker(waker);
        match Pin::new(&mut self.0).poll(&mut cx) {
            std::task::Poll::Ready(r) => {
                #[cfg(debug_assertions)]
                {
                    self.0 = Box::pin(std::future::poll_fn(|_| {
                        panic!("The result of AsyncTask mustn't be used after it returned")
                    }));
                };
                Some(r)
            }
            std::task::Poll::Pending => None,
        }
    }
}
