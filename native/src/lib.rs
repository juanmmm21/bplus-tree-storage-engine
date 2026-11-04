//! Acelerador opcional del códec de páginas de `bplus-tree-storage-engine`.
//!
//! Contrato de equivalencia con la implementación de referencia en Python
//! (`bplus_tree_storage_engine.pipeline`): estas funciones sólo decodifican
//! páginas **válidas**. Ante cualquier anomalía — tamaño, magic, versión,
//! tipo, checksum, punteros o longitudes fuera de rango, clave vacía —
//! devuelven `None` y el llamador usa el decodificador Python, que es el que
//! produce el error exacto de siempre. Así el camino nativo no puede cambiar
//! ningún resultado ni ningún error: sólo es más rápido en el caso válido.
//!
//! Formato (big-endian): cabecera `>4sBBHHII` = magic "BPT1", versión,
//! tipo de página, nº de celdas, offset de espacio libre, puntero lateral,
//! CRC32 (calculado con el propio campo a cero); tras ella, el array de
//! offsets de celda (`u16`); la celda `i` ocupa `[off[i], off[i-1])` (la
//! primera hasta el final de la página).

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};

const PAGE_SIZE: usize = 4096;
const HEADER_SIZE: usize = 18;
const CHECKSUM_OFFSET: usize = 14;
const MAGIC: &[u8; 4] = b"BPT1";
const FORMAT_VERSION: u8 = 1;
const LEAF: u8 = 3;
const INTERNAL: u8 = 2;

fn be_u16(data: &[u8], at: usize) -> Option<usize> {
    let bytes = data.get(at..at + 2)?;
    Some(u16::from_be_bytes([bytes[0], bytes[1]]) as usize)
}

fn be_u32(data: &[u8], at: usize) -> Option<u32> {
    let bytes = data.get(at..at + 4)?;
    Some(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Valida la cabecera y devuelve `(puntero lateral, rangos de celda)`.
fn cells(data: &[u8], expected_type: u8) -> Option<(u32, Vec<(usize, usize)>)> {
    if data.len() != PAGE_SIZE || &data[0..4] != MAGIC || data[4] != FORMAT_VERSION {
        return None;
    }
    if data[5] != expected_type {
        return None;
    }
    let cell_count = be_u16(data, 6)?;
    let side_pointer = be_u32(data, 10)?;
    let stored = be_u32(data, CHECKSUM_OFFSET)?;
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&data[..CHECKSUM_OFFSET]);
    hasher.update(&[0, 0, 0, 0]);
    hasher.update(&data[CHECKSUM_OFFSET + 4..]);
    if hasher.finalize() != stored {
        return None;
    }
    let mut ranges = Vec::with_capacity(cell_count);
    let mut end = PAGE_SIZE;
    for index in 0..cell_count {
        let start = be_u16(data, HEADER_SIZE + index * 2)?;
        if start > end {
            return None;
        }
        ranges.push((start, end));
        end = start;
    }
    Some((side_pointer, ranges))
}

/// `decode_leaf_cells(page) -> (next_leaf, [(key, value), ...]) | None`
#[pyfunction]
fn decode_leaf_cells<'py>(py: Python<'py>, page: &[u8]) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let Some((side_pointer, ranges)) = cells(page, LEAF) else {
        return Ok(None);
    };
    let mut entries: Vec<Bound<'py, PyTuple>> = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        let cell = &page[start..end];
        let Some(key_len) = be_u16(cell, 0) else { return Ok(None) };
        let key_end = 2 + key_len;
        let Some(value_len) = be_u16(cell, key_end) else { return Ok(None) };
        let value_start = key_end + 2;
        let value_end = value_start + value_len;
        if key_len == 0 || value_end > cell.len() {
            return Ok(None);
        }
        let key = PyBytes::new(py, &cell[2..key_end]);
        let value = PyBytes::new(py, &cell[value_start..value_end]);
        entries.push(PyTuple::new(py, [key.into_any(), value.into_any()])?);
    }
    let list = PyList::new(py, entries)?;
    Ok(Some(PyTuple::new(py, [side_pointer.into_pyobject(py)?.into_any(), list.into_any()])?))
}

/// `decode_internal_cells(page) -> (rightmost_child, [(key, left_child), ...]) | None`
#[pyfunction]
fn decode_internal_cells<'py>(
    py: Python<'py>,
    page: &[u8],
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let Some((side_pointer, ranges)) = cells(page, INTERNAL) else {
        return Ok(None);
    };
    let mut entries: Vec<Bound<'py, PyTuple>> = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        let cell = &page[start..end];
        let Some(key_len) = be_u16(cell, 0) else { return Ok(None) };
        let key_end = 2 + key_len;
        let Some(child) = be_u32(cell, key_end) else { return Ok(None) };
        if key_len == 0 {
            return Ok(None);
        }
        let key = PyBytes::new(py, &cell[2..key_end]);
        entries.push(PyTuple::new(py, [key.into_any(), child.into_pyobject(py)?.into_any()])?);
    }
    let list = PyList::new(py, entries)?;
    Ok(Some(PyTuple::new(py, [side_pointer.into_pyobject(py)?.into_any(), list.into_any()])?))
}

#[pymodule]
fn bplus_tree_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(decode_leaf_cells, module)?)?;
    module.add_function(wrap_pyfunction!(decode_internal_cells, module)?)?;
    module.add("FORMAT", "BPT1/v1")?;
    Ok(())
}
