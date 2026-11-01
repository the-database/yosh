//! A model's precision, and whether it suits the backend and GPU (`--model-info`, `HINT` lines).
//!
//! The precision is read from the ONNX file itself, never from its name: the float type that holds
//! most of the elements of the graph's initializers (the weights). A minimal protobuf reader walks
//! ModelProto → GraphProto.initializer → TensorProto and reads each tensor's dims and data type
//! only; everything else, the weights' bytes included, is seeked over, so a 300 MB model costs a
//! few small reads.
//!
//! [`hint`] is the single source of truth for what is suboptimal where: MangaJaNai GUI shows its
//! text as is.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use serde::Serialize;
use yosh_image::aji::Backend;

/// Float initializer elements by type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Weights {
    pub fp32: u64,
    pub fp16: u64,
    pub bf16: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Fp32,
    Fp16,
    Bf16,
    /// No float initializers, or a tie between two types.
    Unknown,
}

impl Precision {
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Fp16 => "fp16",
            Precision::Bf16 => "bf16",
            Precision::Unknown => "unknown",
        }
    }
}

impl Weights {
    /// The type holding the most elements. A tie (or no float weights at all) is `Unknown`, so no
    /// hint is drawn from it.
    pub fn precision(&self) -> Precision {
        let all = [(self.fp32, Precision::Fp32), (self.fp16, Precision::Fp16), (self.bf16, Precision::Bf16)];
        let max = all.iter().map(|&(n, _)| n).max().unwrap_or(0);
        let mut top = all.iter().filter(|&&(n, _)| n == max);
        match (max, top.next(), top.next()) {
            (0, ..) | (_, _, Some(_)) => Precision::Unknown,
            (_, Some(&(_, p)), None) => p,
            (_, None, None) => Precision::Unknown,
        }
    }
}

/// The hint for a `precision` model on `backend`, on a GPU named `gpu` with CUDA
/// `compute_capability` (TensorRT, TensorRT-RTX; ignored on DirectML), or `None` when the model
/// suits it. Without a compute capability on TensorRT/TensorRT-RTX (no device found) there is no
/// hint.
pub fn hint(backend: Backend, compute_capability: Option<(i32, i32)>, gpu: &str, precision: Precision) -> Option<String> {
    use Precision::*;
    if !backend.is_cuda() {
        return match precision {
            Fp32 => Some(format!("fp32 model on {gpu}: an fp16 export typically runs faster on DirectML.")),
            Bf16 => Some("DirectML can't run bf16 models; use an fp16 or fp32 export.".into()),
            Fp16 | Unknown => None,
        };
    }
    let (major, _) = compute_capability?;
    match (precision, major >= 8) {
        (Fp32, true) => Some(format!("fp32 model on {gpu}: a bf16 or fp16 export typically runs about 1.5-2x faster.")),
        (Fp32, false) => Some(format!("fp32 model on {gpu}: an fp16 export typically runs about 2x faster.")),
        (Bf16, false) => Some(format!("{gpu} has no bf16 support; use an fp16 or fp32 export.")),
        (Bf16, true) | (Fp16, _) | (Unknown, _) => None,
    }
}

/// The float initializer elements of the ONNX model at `path`.
pub fn model_weights(path: &Path) -> Result<Weights, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    onnx_weights(file).map_err(|e| format!("{}: {e}", path.display()))
}

// ---- the protobuf reader ---------------------------------------------------------------------

// onnx.proto field numbers.
const MODEL_GRAPH: u32 = 7;
const GRAPH_INITIALIZER: u32 = 5;
const TENSOR_DIMS: u32 = 1;
const TENSOR_DATA_TYPE: u32 = 2;
// TensorProto.DataType.
const FLOAT: i32 = 1;
const FLOAT16: i32 = 10;
const BFLOAT16: i32 = 16;

// Wire types.
const VARINT: u8 = 0;
const I64: u8 = 1;
const LEN: u8 = 2;
const SGROUP: u8 = 3;
const EGROUP: u8 = 4;
const I32: u8 = 5;

/// How deeply groups may nest before the input counts as malformed.
const MAX_DEPTH: u32 = 64;

/// Read the float initializer elements of a serialized ModelProto.
pub fn onnx_weights<R: Read + Seek>(mut r: R) -> Result<Weights, String> {
    let len = r.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    r.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut pb = Pb { r: BufReader::new(r), pos: 0 };
    let mut w = Weights::default();
    while pb.pos < len {
        let (field, wire) = pb.tag()?;
        if field == MODEL_GRAPH && wire == LEN {
            let end = pb.len_end(len)?;
            graph(&mut pb, end, &mut w)?;
        } else {
            pb.skip(field, wire, len, 0)?;
        }
    }
    Ok(w)
}

fn graph<R: Read + Seek>(pb: &mut Pb<R>, end: u64, w: &mut Weights) -> Result<(), String> {
    while pb.pos < end {
        let (field, wire) = pb.tag()?;
        if field == GRAPH_INITIALIZER && wire == LEN {
            let tend = pb.len_end(end)?;
            let (elements, data_type) = tensor(pb, tend)?;
            match data_type {
                FLOAT => w.fp32 = w.fp32.saturating_add(elements),
                FLOAT16 => w.fp16 = w.fp16.saturating_add(elements),
                BFLOAT16 => w.bf16 = w.bf16.saturating_add(elements),
                _ => {}
            }
        } else {
            pb.skip(field, wire, end, 0)?;
        }
    }
    pb.at(end)
}

/// A TensorProto's element count (the product of its dims; a scalar has none and one element) and
/// data type.
fn tensor<R: Read + Seek>(pb: &mut Pb<R>, end: u64) -> Result<(u64, i32), String> {
    let mut elements: u64 = 1;
    let mut data_type = 0;
    let mut dim = |d: u64| elements = elements.saturating_mul(if (d as i64) < 0 { 0 } else { d });
    while pb.pos < end {
        match pb.tag()? {
            (TENSOR_DIMS, VARINT) => dim(pb.varint()?),
            (TENSOR_DIMS, LEN) => {
                let dend = pb.len_end(end)?;
                while pb.pos < dend {
                    dim(pb.varint()?);
                }
                pb.at(dend)?;
            }
            (TENSOR_DATA_TYPE, VARINT) => data_type = pb.varint()? as i32,
            (field, wire) => pb.skip(field, wire, end, 0)?,
        }
    }
    pb.at(end)?;
    Ok((elements, data_type))
}

/// A protobuf stream with its read position.
struct Pb<R> {
    r: BufReader<R>,
    pos: u64,
}

impl<R: Read + Seek> Pb<R> {
    fn byte(&mut self) -> Result<u8, String> {
        let mut b = [0u8];
        self.r.read_exact(&mut b).map_err(|_| "truncated ONNX file".to_string())?;
        self.pos += 1;
        Ok(b[0])
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        for i in 0..10 {
            let b = self.byte()?;
            v |= u64::from(b & 0x7f) << (7 * i);
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err("not an ONNX file (malformed varint)".into())
    }

    /// A field's number and wire type.
    fn tag(&mut self) -> Result<(u32, u8), String> {
        let key = self.varint()?;
        let field = key >> 3;
        if field == 0 || field > u64::from(u32::MAX >> 3) {
            return Err("not an ONNX file (bad field number)".into());
        }
        Ok((field as u32, (key & 7) as u8))
    }

    /// A length-delimited field's end, which must lie within `limit`.
    fn len_end(&mut self, limit: u64) -> Result<u64, String> {
        let n = self.varint()?;
        match self.pos.checked_add(n) {
            Some(end) if end <= limit => Ok(end),
            _ => Err("truncated ONNX file".into()),
        }
    }

    fn seek_to(&mut self, to: u64) -> Result<(), String> {
        let by = i64::try_from(to - self.pos).map_err(|_| "truncated ONNX file".to_string())?;
        self.r.seek_relative(by).map_err(|e| e.to_string())?;
        self.pos = to;
        Ok(())
    }

    /// Skip `n` bytes, which must lie within `limit`.
    fn skip_bytes(&mut self, n: u64, limit: u64) -> Result<(), String> {
        match self.pos.checked_add(n) {
            Some(end) if end <= limit => self.seek_to(end),
            _ => Err("truncated ONNX file".into()),
        }
    }

    /// Skip the value of a field whose tag was just read, staying within `limit`. A group is skipped
    /// through its matching end tag.
    fn skip(&mut self, field: u32, wire: u8, limit: u64, depth: u32) -> Result<(), String> {
        match wire {
            VARINT => self.varint().map(drop),
            I64 => self.skip_bytes(8, limit),
            I32 => self.skip_bytes(4, limit),
            LEN => {
                let end = self.len_end(limit)?;
                self.seek_to(end)
            }
            SGROUP => {
                if depth >= MAX_DEPTH {
                    return Err("not an ONNX file (groups nested too deeply)".into());
                }
                loop {
                    if self.pos >= limit {
                        return Err("truncated ONNX file".into());
                    }
                    match self.tag()? {
                        (f, EGROUP) if f == field => return Ok(()),
                        (_, EGROUP) => return Err("not an ONNX file (mismatched group)".into()),
                        (f, w) => self.skip(f, w, limit, depth + 1)?,
                    }
                }
            }
            _ => Err(format!("not an ONNX file (wire type {wire})")),
        }
    }

    /// The read must have ended exactly at a message's end.
    fn at(&self, end: u64) -> Result<(), String> {
        if self.pos == end { Ok(()) } else { Err("not an ONNX file (field overruns its message)".into()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ---- a tiny protobuf writer for synthetic models ----

    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }

    fn tag(field: u32, wire: u8, out: &mut Vec<u8>) {
        varint(u64::from(field) << 3 | u64::from(wire), out);
    }

    fn len_field(field: u32, body: &[u8], out: &mut Vec<u8>) {
        tag(field, LEN, out);
        varint(body.len() as u64, out);
        out.extend_from_slice(body);
    }

    fn varint_field(field: u32, v: u64, out: &mut Vec<u8>) {
        tag(field, VARINT, out);
        varint(v, out);
    }

    /// A TensorProto with dims (packed or not), a data type, a name and `raw` bytes of data.
    fn tensor(dims: &[i64], data_type: i32, packed: bool, raw: usize) -> Vec<u8> {
        let mut t = Vec::new();
        if packed {
            let mut d = Vec::new();
            for &x in dims {
                varint(x as u64, &mut d);
            }
            len_field(TENSOR_DIMS, &d, &mut t);
        } else {
            for &x in dims {
                varint_field(TENSOR_DIMS, x as u64, &mut t);
            }
        }
        varint_field(TENSOR_DATA_TYPE, data_type as u64, &mut t);
        len_field(8, b"weight", &mut t); // name
        len_field(9, &vec![0xAB; raw], &mut t); // raw_data
        t
    }

    fn model(initializers: &[Vec<u8>]) -> Vec<u8> {
        let mut g = Vec::new();
        len_field(1, b"a node, skipped", &mut g); // node
        len_field(2, b"graph", &mut g); // name
        for t in initializers {
            len_field(GRAPH_INITIALIZER, t, &mut g);
        }
        let mut m = Vec::new();
        varint_field(1, 8, &mut m); // ir_version
        len_field(8, b"\x0a\x00\x10\x11", &mut m); // opset_import
        len_field(2, b"pytorch", &mut m); // producer_name
        len_field(MODEL_GRAPH, &g, &mut m);
        m
    }

    fn weights(bytes: Vec<u8>) -> Result<Weights, String> {
        onnx_weights(Cursor::new(bytes))
    }

    #[test]
    fn counts_float_initializers_by_type() {
        let m = model(&[
            tensor(&[64, 3, 3, 3], FLOAT16, true, 64 * 27 * 2),
            tensor(&[64], FLOAT16, false, 128),
            tensor(&[10, 10], FLOAT, false, 400),
            tensor(&[4, 4], BFLOAT16, true, 32),
            tensor(&[5], 7, true, 40), // INT64: not a weight
            tensor(&[], FLOAT, true, 4), // a scalar
        ]);
        assert_eq!(weights(m).unwrap(), Weights { fp32: 101, fp16: 64 * 27 + 64, bf16: 16 });
    }

    #[test]
    fn packed_and_unpacked_dims_agree_including_big_ones() {
        let dims = [1, 300, 1 << 20, 129];
        let packed = weights(model(&[tensor(&dims, FLOAT, true, 0)])).unwrap();
        let unpacked = weights(model(&[tensor(&dims, FLOAT, false, 0)])).unwrap();
        assert_eq!(packed, unpacked);
        assert_eq!(packed.fp32, 300 * (1 << 20) * 129);
    }

    #[test]
    fn skips_every_wire_type() {
        let mut t = Vec::new();
        tag(20, I64, &mut t);
        t.extend_from_slice(&[1; 8]);
        tag(21, I32, &mut t);
        t.extend_from_slice(&[2; 4]);
        varint_field(22, u64::MAX, &mut t); // a 10-byte varint
        // A group holding a varint, a nested group and a length-delimited field.
        tag(23, SGROUP, &mut t);
        varint_field(1, 5, &mut t);
        tag(2, SGROUP, &mut t);
        tag(2, EGROUP, &mut t);
        len_field(3, b"xyz", &mut t);
        tag(23, EGROUP, &mut t);
        t.extend_from_slice(&tensor(&[2, 3], FLOAT16, true, 12));

        let mut m = Vec::new();
        tag(30, SGROUP, &mut m); // an unknown top-level group
        tag(30, EGROUP, &mut m);
        tag(31, I32, &mut m);
        m.extend_from_slice(&[0; 4]);
        m.extend_from_slice(&model(&[t]));
        assert_eq!(weights(m).unwrap(), Weights { fp16: 6, ..Weights::default() });
    }

    #[test]
    fn repeated_graph_fields_merge() {
        let mut m = model(&[tensor(&[3], FLOAT, true, 12)]);
        m.extend_from_slice(&model(&[tensor(&[5], BFLOAT16, true, 10)]));
        assert_eq!(weights(m).unwrap(), Weights { fp32: 3, bf16: 5, ..Weights::default() });
    }

    #[test]
    fn rejects_malformed_input() {
        let good = model(&[tensor(&[8, 8], FLOAT, true, 256)]);
        // Cut anywhere inside: never a panic, and an error unless the cut leaves whole fields.
        for n in 1..good.len() {
            let _ = weights(good[..n].to_vec());
        }
        assert!(weights(good[..good.len() - 1].to_vec()).is_err());
        assert!(weights(b"\x00\x01".to_vec()).is_err(), "field number 0");
        assert!(weights(b"\x0e".to_vec()).is_err(), "wire type 6");
        assert!(weights(vec![0xff; 11]).is_err(), "endless varint");
        let mut g = Vec::new();
        tag(5, SGROUP, &mut g);
        tag(6, EGROUP, &mut g);
        assert!(weights(g).is_err(), "mismatched group");
        assert_eq!(weights(Vec::new()).unwrap(), Weights::default());
    }

    #[test]
    fn precision_is_the_majority_type() {
        let w = |fp32, fp16, bf16| Weights { fp32, fp16, bf16 }.precision();
        assert_eq!(w(10, 0, 0), Precision::Fp32);
        assert_eq!(w(10, 1000, 3), Precision::Fp16);
        assert_eq!(w(400, 0, 16_000_000), Precision::Bf16);
        assert_eq!(w(0, 0, 0), Precision::Unknown);
        assert_eq!(w(5, 5, 0), Precision::Unknown);
        assert_eq!(w(1, 5, 5), Precision::Unknown);
    }

    #[test]
    fn hint_rules() {
        use Precision::*;
        let gpu = "GPU X";
        let (trt, rtx, dml) = (Backend::TensorRt, Backend::TensorRtRtx, Backend::DirectMl);
        let ampere_up = [Some((8, 0)), Some((8, 6)), Some((8, 9)), Some((9, 0)), Some((12, 0))];
        let turing_down = [Some((7, 5)), Some((7, 0)), Some((6, 1))];
        for b in [trt, rtx] {
            for cc in ampere_up {
                assert_eq!(
                    hint(b, cc, gpu, Fp32).as_deref(),
                    Some("fp32 model on GPU X: a bf16 or fp16 export typically runs about 1.5-2x faster.")
                );
                for p in [Fp16, Bf16, Unknown] {
                    assert_eq!(hint(b, cc, gpu, p), None, "{b:?} {cc:?} {p:?}");
                }
            }
            for cc in turing_down {
                assert_eq!(
                    hint(b, cc, gpu, Fp32).as_deref(),
                    Some("fp32 model on GPU X: an fp16 export typically runs about 2x faster.")
                );
                assert_eq!(hint(b, cc, gpu, Bf16).as_deref(), Some("GPU X has no bf16 support; use an fp16 or fp32 export."));
                assert_eq!(hint(b, cc, gpu, Fp16), None);
                assert_eq!(hint(b, cc, gpu, Unknown), None);
            }
            // No device, no compute capability: nothing to say.
            for p in [Fp32, Fp16, Bf16, Unknown] {
                assert_eq!(hint(b, None, gpu, p), None);
            }
        }
        for cc in [None, Some((12, 0)), Some((7, 5))] {
            assert_eq!(
                hint(dml, cc, gpu, Fp32).as_deref(),
                Some("fp32 model on GPU X: an fp16 export typically runs faster on DirectML.")
            );
            assert_eq!(hint(dml, cc, gpu, Bf16).as_deref(), Some("DirectML can't run bf16 models; use an fp16 or fp32 export."));
            assert_eq!(hint(dml, cc, gpu, Fp16), None);
            assert_eq!(hint(dml, cc, gpu, Unknown), None);
        }
    }
}
