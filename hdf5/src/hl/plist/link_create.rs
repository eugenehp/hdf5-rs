//! Link creation property list.

use std::fmt::{self, Debug};
use std::ops::Deref;

use crate::class::ObjectClass;
use crate::error::Result;
use crate::h5i::H5I_type_t;
use crate::handle::Handle;
use crate::hl::plist::{PlistState, PropertyList};

pub(crate) const PROPERTY_NAMES: &[&str] = &["create_intermediate_group", "char_encoding"];

/// Character encoding of link names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CharEncoding {
    #[default]
    Ascii,
    Utf8,
}

/// The data carried by a link-creation property list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) struct LinkCreateData {
    pub create_intermediate_group: bool,
    pub char_encoding: CharEncoding,
}

/// Link creation property list.
#[repr(transparent)]
#[derive(Clone)]
pub struct LinkCreate(Handle);

impl ObjectClass for LinkCreate {
    const NAME: &'static str = "link create property list";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_type_t::H5I_GENPROP_LST];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }
}

impl Debug for LinkCreate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for LinkCreate {
    type Target = PropertyList;

    fn deref(&self) -> &PropertyList {
        unsafe { self.transmute() }
    }
}

impl PartialEq for LinkCreate {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for LinkCreate {}

impl Default for LinkCreate {
    fn default() -> Self {
        Self::try_new().unwrap()
    }
}

impl LinkCreate {
    pub(crate) fn from_data(data: LinkCreateData) -> Self {
        Self(PropertyList::from_state(PlistState::LinkCreate(data)).0)
    }

    pub(crate) fn data(&self) -> LinkCreateData {
        match self.0.plist_state() {
            Some(PlistState::LinkCreate(d)) => *d,
            _ => LinkCreateData::default(),
        }
    }

    pub fn try_new() -> Result<Self> {
        Ok(Self::from_data(LinkCreateData::default()))
    }

    pub fn copy(&self) -> Self {
        Self::from_data(self.data())
    }

    pub fn build() -> LinkCreateBuilder {
        LinkCreateBuilder::new()
    }

    pub fn get_create_intermediate_group(&self) -> Result<bool> {
        Ok(self.create_intermediate_group())
    }

    pub fn get_char_encoding(&self) -> Result<CharEncoding> {
        Ok(self.char_encoding())
    }

    pub fn create_intermediate_group(&self) -> bool {
        self.data().create_intermediate_group
    }

    pub fn char_encoding(&self) -> CharEncoding {
        self.data().char_encoding
    }
}

/// Builder for link creation property lists.
#[derive(Clone, Debug, Default)]
pub struct LinkCreateBuilder {
    data: LinkCreateData,
}

impl LinkCreateBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_plist(plist: &LinkCreate) -> Result<Self> {
        Ok(Self { data: plist.data() })
    }

    pub fn create_intermediate_group(&mut self, create: bool) -> &mut Self {
        self.data.create_intermediate_group = create;
        self
    }

    pub fn char_encoding(&mut self, encoding: CharEncoding) -> &mut Self {
        self.data.char_encoding = encoding;
        self
    }

    pub fn apply(&self, plist: &mut LinkCreate) -> Result<()> {
        *plist = LinkCreate::from_data(self.data);
        Ok(())
    }

    pub fn finish(&self) -> Result<LinkCreate> {
        Ok(LinkCreate::from_data(self.data))
    }
}
