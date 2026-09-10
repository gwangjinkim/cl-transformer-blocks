//! Narrow owned-pointer ABI to the same tokenizer engine used by Hugging Face.
use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{catch_unwind, AssertUnwindSafe};
use tokenizers::{EncodeInput, PaddingDirection, PostProcessor, Tokenizer, TruncationParams};
thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }
fn guarded<T>(failure: T, f: impl FnOnce() -> Result<T, String>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        result => {
            let message = match result { Ok(Err(e)) => e, _ => "Tokenizer panicked".into() };
            ERROR.with(|slot| *slot.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap());
            failure
        }
    }
}
unsafe fn string<'a>(p: *const c_char) -> Result<&'a str, String> {
    if p.is_null() { return Err("Null string".into()); }
    CStr::from_ptr(p).to_str().map_err(|e| e.to_string())
}
#[no_mangle]
pub extern "C" fn tb_tokenizer_error() -> *const c_char {
    ERROR.with(|slot| slot.borrow().as_ptr())
}
#[no_mangle]
pub unsafe extern "C" fn tb_tokenizer_load(path: *const c_char) -> *mut Tokenizer {
    guarded(std::ptr::null_mut(), || {
        let mut tokenizer = Tokenizer::from_file(string(path)?).map_err(|e| e.to_string())?;
        // Per-call tokenization does not inherit training-time padding/truncation.
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(tokenizer)))
    })
}
#[no_mangle]
pub unsafe extern "C" fn tb_tokenizer_free(p: *mut Tokenizer) {
    if !p.is_null() { drop(Box::from_raw(p)); }
}
#[no_mangle]
pub unsafe extern "C" fn tb_tokenizer_from_bytes(
    data: *const u8, length: usize, vocabulary_size: u32,
) -> *mut Tokenizer {
    guarded(std::ptr::null_mut(), || {
        if data.is_null() || length == 0 { return Err("Empty tokenizer data".into()); }
        let mut tokenizer = Tokenizer::from_bytes(std::slice::from_raw_parts(data, length))
            .map_err(|e| e.to_string())?;
        let vocabulary = tokenizer.get_vocab(true);
        if vocabulary.is_empty() || vocabulary.values().any(|id| *id >= vocabulary_size) {
            return Err("Tokenizer vocabulary IDs exceed model vocabulary_size".into());
        }
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(tokenizer)))
    })
}
#[no_mangle]
pub unsafe extern "C" fn tb_tokenizer_encode(p: *const Tokenizer, text: *const c_char, special: i32) -> *mut c_char {
    guarded(std::ptr::null_mut(), || {
        let tokenizer = p.as_ref().ok_or("Null tokenizer")?;
        let encoding = tokenizer.encode(string(text)?, special != 0).map_err(|e| e.to_string())?;
        Ok(CString::new(serde_json::to_string(encoding.get_ids()).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?.into_raw())
    })
}
#[no_mangle]
pub unsafe extern "C" fn tb_tokenizer_encode_batch(p: *const Tokenizer, request: *const c_char) -> *mut c_char {
    guarded(std::ptr::null_mut(), || {
        let tokenizer = p.as_ref().ok_or("Null tokenizer")?;
        let mut request: serde_json::Value = serde_json::from_str(string(request)?).map_err(|e| e.to_string())?;
        let texts: Vec<String> = serde_json::from_value(request["texts"].take()).map_err(|e| e.to_string())?;
        let pairs: Option<Vec<String>> = serde_json::from_value(request["text_pairs"].take()).map_err(|e| e.to_string())?;
        if texts.is_empty() || pairs.as_ref().is_some_and(|pairs| pairs.len() != texts.len()) {
            return Err("Expected a nonempty batch with matching text pairs".into());
        }
        let special = request["add_special_tokens"].as_bool().ok_or("Invalid special-token flag")?;
        let truncate = request["truncation"].as_bool().ok_or("Invalid truncation flag")?;
        let maximum = request["max_length"].as_u64()
            .and_then(|n| usize::try_from(n).ok()).filter(|n| *n > 0).ok_or("Invalid maximum length")?;
        let fixed_padding = match request["padding"].as_str() {
            Some("longest") => false, Some("max-length") => true,
            _ => return Err("Invalid padding policy".into()),
        };
        let pad_id = if request["pad_token_id"].is_null() { None } else {
            Some(request["pad_token_id"].as_u64().and_then(|n| u32::try_from(n).ok())
                .ok_or("Invalid padding token ID")?)
        };
        // Borrow the resident engine unless truncation needs a private configuration.
        let mut configured;
        let engine = if truncate {
            let processor = tokenizer.get_post_processor();
            let reserved = processor.map_or(0, |p| p.added_tokens(pairs.is_some()));
            if special && maximum < reserved {
                return Err("Maximum length cannot fit the tokenizer's special tokens".into());
            }
            configured = tokenizer.clone();
            // Tokenizers 0.23.2's setter subtracts single-sequence special tokens
            // even when add_special_tokens=false. Avoid unsigned underflow there.
            let setter_maximum = maximum.max(processor.map_or(0, |p| p.added_tokens(false)));
            configured.with_truncation(Some(TruncationParams {
                max_length: setter_maximum, ..Default::default()
            })).map_err(|e| e.to_string())?;
            configured.get_truncation_mut().ok_or("Missing truncation configuration")?.max_length = maximum;
            &configured
        } else { tokenizer };
        let inputs: Vec<EncodeInput> = texts.iter().enumerate().map(|(i, text)| {
            match &pairs {
                Some(pairs) => (text.as_str(), pairs[i].as_str()).into(),
                None => text.as_str().into(),
            }
        }).collect();
        let mut encodings = engine.encode_batch(inputs, special).map_err(|e| e.to_string())?;
        let longest = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        if longest > maximum || encodings.iter().any(|e| e.is_empty()) {
            return Err("Batch contains an empty encoding or exceeds maximum length".into());
        }
        let width = if fixed_padding { maximum } else { longest };
        for encoding in &mut encodings {
            if encoding.get_ids().len() < width {
                encoding.pad(width, pad_id.ok_or("Padding requires a pad token ID")?, 0, "", PaddingDirection::Right);
            }
        }
        let result = serde_json::json!({
            "input_ids": encodings.iter().map(|e| e.get_ids()).collect::<Vec<_>>(),
            "attention_mask": encodings.iter().map(|e| e.get_attention_mask()).collect::<Vec<_>>(),
            "token_type_ids": encodings.iter().map(|e| e.get_type_ids()).collect::<Vec<_>>()
        });
        Ok(CString::new(result.to_string()).map_err(|e| e.to_string())?.into_raw())
    })
}
#[no_mangle]
pub unsafe extern "C" fn tb_tokenizer_decode(p: *const Tokenizer, ids: *const c_char, skip: i32) -> *mut c_char {
    guarded(std::ptr::null_mut(), || {
        let tokenizer = p.as_ref().ok_or("Null tokenizer")?;
        let ids: Vec<u32> = serde_json::from_str(string(ids)?).map_err(|e| e.to_string())?;
        let text = tokenizer.decode(&ids, skip != 0).map_err(|e| e.to_string())?;
        Ok(CString::new(text).map_err(|e| e.to_string())?.into_raw())
    })
}
#[no_mangle]
pub unsafe extern "C" fn tb_tokenizer_string_free(p: *mut c_char) {
    if !p.is_null() { drop(CString::from_raw(p)); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_truncation_and_failure_leave_resident_tokenizer_unchanged() {
        let tokenizer = Tokenizer::from_bytes(br#"{
            "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
            "normalizer":null, "pre_tokenizer":{"type":"WhitespaceSplit"},
            "post_processor":{"type":"BertProcessing","sep":["[SEP]",3],"cls":["[CLS]",2]},
            "decoder":null, "model":{"type":"WordLevel","unk_token":"[UNK]",
            "vocab":{"[UNK]":0,"[PAD]":1,"[CLS]":2,"[SEP]":3,"Lisp":4}}
        }"#).unwrap();
        let mut request = serde_json::json!({"texts":["Lisp Lisp"], "text_pairs":null,
            "add_special_tokens":false, "truncation":true, "max_length":1,
            "padding":"longest", "pad_token_id":1});
        unsafe {
            let json = CString::new(request.to_string()).unwrap();
            let output = tb_tokenizer_encode_batch(&tokenizer, json.as_ptr());
            assert!(!output.is_null(), "{}", string(tb_tokenizer_error()).unwrap());
            let result: serde_json::Value = serde_json::from_str(string(output).unwrap()).unwrap();
            tb_tokenizer_string_free(output);
            assert_eq!(result["input_ids"], serde_json::json!([[4]]));
            request["add_special_tokens"] = true.into();
            let json = CString::new(request.to_string()).unwrap();
            assert!(tb_tokenizer_encode_batch(&tokenizer, json.as_ptr()).is_null());
            let text = CString::new("Lisp Lisp").unwrap();
            let output = tb_tokenizer_encode(&tokenizer, text.as_ptr(), 1);
            assert!(!output.is_null());
            assert_eq!(string(output).unwrap(), "[2,4,4,3]");
            tb_tokenizer_string_free(output);
        }
        assert!(tokenizer.get_truncation().is_none());
        assert!(tokenizer.get_padding().is_none());
    }
}
