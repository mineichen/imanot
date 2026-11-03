use std::rc::Rc;

use imanot::Tool;

/// Creates a fresh tool each time an entry gets bound to a slot.
pub type ToolFactory = Rc<dyn Fn() -> Box<dyn Tool>>;

/// The registry of available tools and their display names.
///
/// This owns the tool "dictionary" (name -> creator) and tracks which entries
/// are currently bound to the primary and secondary slots. The actual active
/// tool instances live in [`imanot::Tools`]; this registry only decides which
/// tool gets created for it.
pub struct ToolRegistry {
    tools: Vec<(String, ToolFactory)>,
    primary_idx: usize,
    secondary_idx: usize,
}

impl ToolRegistry {
    pub fn new(tools: Vec<(String, ToolFactory)>) -> Self {
        let primary_idx = 0;
        let secondary_idx = if tools.len() > 1 { 1 } else { 0 };
        Self {
            tools,
            primary_idx,
            secondary_idx,
        }
    }

    pub fn primary_idx(&self) -> usize {
        self.primary_idx
    }
    pub fn secondary_idx(&self) -> usize {
        self.secondary_idx
    }

    pub fn name(&self, idx: usize) -> &str {
        self.tools
            .get(idx)
            .map(|x| x.0.as_str())
            .unwrap_or("<unknown>")
    }

    pub fn new_primary(&self) -> Box<dyn Tool> {
        (self.tools[self.primary_idx].1)()
    }
    pub fn new_secondary(&self) -> Box<dyn Tool> {
        (self.tools[self.secondary_idx].1)()
    }

    /// Set the primary slot index. Returns a new tool if it changed.
    pub fn set_primary_idx(&mut self, idx: usize) -> Option<Box<dyn Tool>> {
        (idx < self.tools.len() && idx != self.primary_idx).then(|| {
            self.primary_idx = idx;
            (self.tools[idx].1)()
        })
    }
    /// Set the secondary slot index. Returns a new tool if it changed.
    pub fn set_secondary_idx(&mut self, idx: usize) -> Option<Box<dyn Tool>> {
        (idx < self.tools.len() && idx != self.secondary_idx).then(|| {
            self.secondary_idx = idx;
            (self.tools[idx].1)()
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.iter().map(|(name, _)| name.as_str())
    }
}

impl<'a> From<&'a crate::config::Config> for ToolRegistry {
    fn from(config: &'a crate::config::Config) -> Self {
        ToolRegistry::new(super::default_tools(config))
    }
}
