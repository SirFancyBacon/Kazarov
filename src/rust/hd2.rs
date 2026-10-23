//! Unpacks Dark Cloud's disc, DATA.DAT and the DATA.HD2 index of its files, into the retail tree
//! the Chronicle port reads: every file the game can reach, a copy of the index, the fixes its
//! releases need (written beside the originals or under `normalized/`) and `languages.json`.
use super::archive::{
    Extent, ReadOnly, SECTOR, copy_extent, directory_file, disk_name, extent_matches, folded,
    image_extent, le32, nul_name, safe_relative,
};
use super::progress::Phase;
use super::{CoreError, CoreResult};
use gpt_disk_io::{BlockIoAdapter, gpt_disk_types::BlockSize};
use iso9660::{find_file, mount};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::{self, File};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

fn fail(message: impl Into<String>) -> CoreError {
    CoreError(message.into())
}

fn le16(bytes: &[u8], offset: usize) -> usize {
    usize::from(bytes[offset]) | usize::from(bytes[offset + 1]) << 8
}

fn put32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn sectors_for(size: u64) -> u64 {
    size.div_ceil(SECTOR)
}

/// DATA.DAT and DATA.HD2, as files of their own or extents of a disc image.
struct Disc {
    source: PathBuf,
    image: bool,
    dat: Extent,
    hd2: Extent,
}

fn open_image(image: &Path) -> CoreResult<Disc> {
    let block_size =
        BlockSize::new(SECTOR as u32).ok_or_else(|| fail("invalid ISO sector size"))?;
    let mut io = BlockIoAdapter::new(ReadOnly(File::open(image)?), block_size);
    let volume = mount(&mut io, 0)
        .map_err(|cause| fail(format!("{}: ISO mount: {cause}", image.display())))?;
    if u64::from(volume.logical_block_size) != SECTOR {
        return Err(fail(format!(
            "{}: unsupported ISO logical block size",
            image.display()
        )));
    }
    let mut required = |name: &str| -> CoreResult<Extent> {
        let entry = find_file(&mut io, &volume, name)
            .map_err(|cause| fail(format!("{}: ISO {name}: {cause}", image.display())))?;
        if entry.flags.directory {
            return Err(fail(format!("ISO {name} is a directory")));
        }
        image_extent(image, &entry)
    };
    Ok(Disc {
        source: image.to_path_buf(),
        image: true,
        dat: required("/DATA.DAT")?,
        hd2: required("/DATA.HD2")?,
    })
}

fn open_disc(source: &Path) -> CoreResult<Disc> {
    if source.is_dir() {
        Ok(Disc {
            source: source.to_path_buf(),
            image: false,
            dat: directory_file(source, "DATA.DAT")?,
            hd2: directory_file(source, "DATA.HD2")?,
        })
    } else if source.is_file() {
        open_image(source)
    } else {
        Err(fail(format!(
            "{} is neither a disc image nor a directory",
            source.display()
        )))
    }
}

/// One file of DATA.HD2: its name as the disc spells it and the path it is extracted to.
#[derive(Clone)]
struct Record {
    name: Vec<u8>,
    path: String,
    size: u32,
    sector: u32,
    sectors: u32,
}

impl Record {
    fn display(&self) -> String {
        String::from_utf8_lossy(&self.name).into_owned()
    }
}

/// DATA.HD2 as PAL's InitCDFile reads it: 32-byte records (name offset, three unused words,
/// offset, size, sector, sector count) followed by the names, which the first record's name
/// offset counts.
fn parse_index(hd2: &[u8]) -> CoreResult<Vec<Record>> {
    if hd2.len() < 32 {
        return Err(fail(format!(
            "DATA.HD2 is {} bytes, too short for one record",
            hd2.len()
        )));
    }
    let count = le32(hd2)? as usize >> 5;
    if count == 0 || count * 32 > hd2.len() {
        return Err(fail(format!(
            "DATA.HD2 claims {count} records but holds {} bytes",
            hd2.len()
        )));
    }
    let mut records = Vec::with_capacity(count);
    for index in 0..count {
        let raw = &hd2[index * 32..index * 32 + 32];
        let name_offset = le32(raw)? as usize;
        if name_offset >= hd2.len() {
            return Err(fail(format!(
                "DATA.HD2 record {index}: name offset {name_offset} is past the end of the index"
            )));
        }
        let name = nul_name(hd2, name_offset, &format!("DATA.HD2 record {index}"))?;
        // The retail index ends with one all-zero record, which the game's linear search never
        // matches; the count above includes it.
        if name.is_empty() && index > 0 && le32(&raw[20..])? == 0 {
            break;
        }
        let path = disk_name(&folded(&name));
        if !safe_relative(&path) {
            return Err(fail(format!(
                "DATA.HD2 record {index}: unusable path \"{}\"",
                String::from_utf8_lossy(&name)
            )));
        }
        let record = Record {
            name,
            path,
            size: le32(&raw[20..])?,
            sector: le32(&raw[24..])?,
            sectors: le32(&raw[28..])?,
        };
        if record.size > 0x7FFF_FFFF || record.sector > 0x7FFF_FFFF {
            return Err(fail(format!(
                "DATA.HD2 record {index} ({}): size {} or sector {} is negative",
                record.display(),
                record.size,
                record.sector
            )));
        }
        records.push(record);
    }
    Ok(records)
}

/// The first record of each path, as PAL's linear search finds, and how many later ones repeat one.
fn reachable<'a>(records: &'a [Record], phase: &Phase) -> (Vec<&'a Record>, u64) {
    let mut seen = HashSet::new();
    let mut found = Vec::new();
    let mut duplicates = 0;
    for record in records {
        if seen.insert(record.path.as_str()) {
            found.push(record);
        } else {
            duplicates += 1;
            phase.log(format!(
                "warning: {} is listed again at sector {}; the game only ever finds the first",
                record.display(),
                record.sector
            ));
        }
    }
    (found, duplicates)
}

/// The records of files the game can load, and how many others the index lists. The game names
/// every file it loads in ASCII; the disc's other names are its developers' leftovers (a note, a
/// shortcut and empty files labelling map folders), which nothing on the disc refers to.
fn loadable(records: Vec<Record>) -> (Vec<Record>, u64) {
    let (loadable, other): (Vec<_>, Vec<_>) = records
        .into_iter()
        .partition(|record| record.name.is_ascii());
    (loadable, other.len() as u64)
}

fn check_fits(records: &[Record], dat_size: u64, phase: &Phase) -> CoreResult<u64> {
    let mut warnings = 0;
    for record in records {
        let end = u64::from(record.sector) * SECTOR + u64::from(record.size);
        if end > dat_size {
            return Err(fail(format!(
                "DATA.DAT is truncated: {} needs bytes up to {end}, DATA.DAT holds {dat_size}",
                record.display()
            )));
        }
        if u64::from(record.sectors) < sectors_for(u64::from(record.size)) {
            warnings += 1;
            phase.log(format!(
                "warning: {} is {} bytes but the index gives it {} sectors",
                record.display(),
                record.size,
                record.sectors
            ));
        }
    }
    Ok(warnings)
}

fn is_current(path: &Path, size: u64) -> bool {
    fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() == size)
}

fn write_file(path: &Path, bytes: &[u8]) -> CoreResult<()> {
    fs::create_dir_all(path.parent().ok_or_else(|| fail("target has no parent"))?)?;
    fs::write(path, bytes)
        .map_err(|error| fail(format!("cannot write {}: {error}", path.display())))
}

/// A PACK member: the game's packs are 76-byte entries (a 64-byte name, the data's offset from the
/// entry, its size and the next entry's distance) each followed by its data.
#[derive(Clone)]
struct Member {
    name: Vec<u8>,
    data: Vec<u8>,
}

fn read_pack(pack: &[u8]) -> CoreResult<Vec<Member>> {
    let mut members = Vec::new();
    let mut entry = 0;
    while entry + 76 <= pack.len() && pack[entry] != 0 {
        let end = pack[entry..entry + 64]
            .iter()
            .position(|&byte| byte == 0)
            .ok_or_else(|| fail(format!("pack entry at {entry} has no terminated name")))?;
        let offset = le32(&pack[entry + 64..])? as usize;
        let size = le32(&pack[entry + 68..])? as usize;
        let next = le32(&pack[entry + 72..])? as usize;
        let left = pack.len() - entry;
        if offset < 76 || offset > left || size > left - offset || next < 76 || next > left {
            return Err(fail(format!("invalid pack entry at {entry}")));
        }
        members.push(Member {
            name: pack[entry..entry + end].to_vec(),
            data: pack[entry + offset..entry + offset + size].to_vec(),
        });
        entry += next;
    }
    Ok(members)
}

fn write_pack(members: &[Member]) -> CoreResult<Vec<u8>> {
    let mut pack = Vec::new();
    for member in members {
        if member.name.len() >= 64 {
            return Err(fail(format!(
                "pack member name is too long: {}",
                String::from_utf8_lossy(&member.name)
            )));
        }
        let entry = pack.len();
        pack.resize(entry + 80, 0);
        pack[entry..entry + member.name.len()].copy_from_slice(&member.name);
        put32(&mut pack, entry + 64, 80);
        put32(&mut pack, entry + 68, member.data.len() as u32);
        pack.extend_from_slice(&member.data);
        pack.resize(pack.len().next_multiple_of(16), 0);
        let next = (pack.len() - entry) as u32;
        put32(&mut pack, entry + 72, next);
    }
    pack.resize(pack.len() + 76, 0);
    Ok(pack)
}

fn find_member<'a>(members: &'a [Member], name: &str) -> Option<&'a Member> {
    members
        .iter()
        .find(|member| member.name.eq_ignore_ascii_case(name.as_bytes()))
}

fn add_pack_alias(members: &mut Vec<Member>, name: &str, source: &str) -> bool {
    if find_member(members, name).is_some() {
        return false;
    }
    let Some(data) = find_member(members, source).map(|member| member.data.clone()) else {
        return false;
    };
    members.push(Member {
        name: name.as_bytes().to_vec(),
        data,
    });
    true
}

/// Where the picture `name` of an IM2 bank lies: its TIM2 file's bytes.
fn find_im2_picture(bank: &[u8], name: &str) -> Option<Range<usize>> {
    if bank.len() < 16 || &bank[..3] != b"IM2" {
        return None;
    }
    let count = le32(&bank[4..]).ok()? as usize;
    if 16 + count * 48 > bank.len() {
        return None;
    }
    for index in 0..count {
        let entry = &bank[16 + index * 48..16 + index * 48 + 48];
        let length = entry[..32].iter().position(|&byte| byte == 0).unwrap_or(32);
        if &entry[..length] != name.as_bytes() {
            continue;
        }
        let offset = le32(&entry[32..]).ok()? as usize;
        if offset + 32 > bank.len() || &bank[offset..offset + 4] != b"TIM2" {
            return None;
        }
        // The picture header's total size is not reliable in the game's images; its parts are.
        let picture = &bank[offset + 16..];
        let size = 16
            + le16(picture, 12)
            + le32(&picture[8..]).ok()? as usize
            + le32(&picture[4..]).ok()? as usize;
        if size > bank.len() - offset {
            return None;
        }
        return Some(offset..offset + size);
    }
    None
}

/// An IM2 bank of the named TIM2 pictures, in order, as the game's banks lay them out.
fn write_im2(pictures: &[(&str, Vec<u8>)]) -> CoreResult<Vec<u8>> {
    let mut bank = vec![0; 16 + pictures.len() * 48];
    bank[..3].copy_from_slice(b"IM2");
    put32(&mut bank, 4, pictures.len() as u32);
    for (index, (name, picture)) in pictures.iter().enumerate() {
        if name.len() >= 32 {
            return Err(fail(format!("IM2 picture name is too long: {name}")));
        }
        bank[16 + index * 48..16 + index * 48 + name.len()].copy_from_slice(name.as_bytes());
        let offset = bank.len() as u32;
        put32(&mut bank, 16 + index * 48 + 32, offset);
        bank.extend_from_slice(picture);
    }
    Ok(bank)
}

/// The colours of an 8-bit TIM2 picture's 256-entry CLUT as RGB, or none.
fn tim2_colours(tim: &[u8]) -> Vec<u32> {
    if tim.len() < 48 || tim[16 + 0x13] != 5 || le16(tim, 16 + 0x0E) != 256 {
        return Vec::new();
    }
    let Ok(pixels) = le32(&tim[16 + 8..]) else {
        return Vec::new();
    };
    let clut = 16 + le16(tim, 16 + 0x0C) + pixels as usize;
    if clut + 1024 > tim.len() {
        return Vec::new();
    }
    tim[clut..clut + 1024]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|colour| u32::from_le_bytes([colour[0], colour[1], colour[2], 0]))
        .collect()
}

fn shared_colours(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x == y).count()
}

/// The dungeon copies weapon icons by palette index from whichever wepicon sheet is loaded into the
/// HUD's itempack: a dungeon-entry pack's on a new floor, then a menu's once the main menu or the
/// quick-change ring closes or a script reloads the item list. So every wepicon sheet of a language
/// has to share itempack's palette, as the American ones do. On the PAL disc only English's
/// dungeon-entry sheets do, and the HUD icon comes out in the wrong colours after any of those; the
/// other languages' fishing and shop sheets are off as well, though only the town shows them. A
/// sheet in another palette takes the dungeon-entry sheet's place: the same icons in itempack's.
fn normalize_icon_sheets(out: &Path, records: &[Record]) -> CoreResult<()> {
    // A sheet in another palette shares a handful of its colours with itempack's.
    const SHARED_PALETTE: usize = 250;
    for record in records {
        let Some(language) = record
            .path
            .strip_prefix("commenu/")
            .and_then(|rest| rest.strip_suffix("/itempack.img"))
        else {
            continue;
        };
        if language.is_empty() || language.contains('/') {
            continue;
        }
        let itempack = fs::read(out.join(&record.path))?;
        let Some(hud) = find_im2_picture(&itempack, "itempack") else {
            continue;
        };
        let hud_colours = tim2_colours(&itempack[hud]);

        let entry_packs = format!("commenu/{language}/dunenter/");
        let mut donor = Vec::new();
        for other in records {
            if !donor.is_empty()
                || !other.path.starts_with(&entry_packs)
                || !other.path.ends_with(".pak")
            {
                continue;
            }
            for member in read_pack(&fs::read(out.join(&other.path))?)? {
                if let Some(icons) = find_im2_picture(&member.data, "wepicon")
                    && shared_colours(&tim2_colours(&member.data[icons.clone()]), &hud_colours)
                        >= SHARED_PALETTE
                {
                    donor = member.data[icons].to_vec();
                    break;
                }
            }
        }
        if donor.is_empty() {
            continue;
        }

        // Whether the bank's wepicon was in another palette and the donor took its place.
        let replace = |bank: &mut Vec<u8>| {
            let Some(icons) = find_im2_picture(bank, "wepicon") else {
                return false;
            };
            if icons.len() != donor.len()
                || shared_colours(&tim2_colours(&bank[icons.clone()]), &hud_colours)
                    >= SHARED_PALETTE
            {
                return false;
            }
            bank[icons].copy_from_slice(&donor);
            true
        };
        // Every image bank and pack of the language, but the leftovers the game never names (_x,
        // x.old).
        let folder = format!("commenu/{language}/");
        for file in records {
            let path = file.path.as_str();
            let name = &path[path.rfind('/').map_or(0, |slash| slash + 1)..];
            let bank = path.ends_with(".img");
            if !path.starts_with(&folder)
                || name.starts_with('_')
                || !(bank || path.ends_with(".pak") || path.ends_with(".pac"))
            {
                continue;
            }
            let mut data = fs::read(out.join(path))?;
            if bank {
                if replace(&mut data) {
                    write_file(&out.join("normalized").join(path), &data)?;
                }
                continue;
            }
            // A .pac that holds no pack is left alone.
            let Ok(mut members) = read_pack(&data) else {
                continue;
            };
            let mut changed = false;
            for member in &mut members {
                changed |= replace(&mut member.data);
            }
            if changed {
                write_file(&out.join("normalized").join(path), &write_pack(&members)?)?;
            }
        }
    }
    Ok(())
}

/// The town system pack: one on NTSC, which PAL split into a common part and one per language.
const SYSTEM_PACK: &str = "gedit/system/esys.pak";
const PAL_COMMON_SYSTEM_PACK: &str = "gedit/system/esys_cmn.pak";

/// Whether a disc's files are NTSC's, by the presence of each system pack.
fn is_ntsc_layout(paths: &HashSet<&str>) -> bool {
    paths.contains(SYSTEM_PACK) && !paths.contains(PAL_COMMON_SYSTEM_PACK)
}

/// PAL split NTSC's town system packs. esys_cmn.pak holds what every language shares, with
/// sys_cmn.img the three pictures of NTSC's system.img that PAL kept; sys_N.img (sys.img for
/// Japanese) holds a language's pause and day-of-adventure pictures, which NTSC packs in
/// esys_N.pak as pause.img and whatsday.img. NTSC's gaiji, fuki256 and syst04 are meswin's, and
/// its skip_bord and second pause PAL dropped. From NTSC 1.02 these are PAL's members and pictures
/// exactly.
fn split_system_packs(out: &Path, paths: &HashSet<&str>) -> CoreResult<()> {
    let picture =
        |members: &[Member], image: &str, name: &str, pack: &str| -> CoreResult<Vec<u8>> {
            find_member(members, image)
                .and_then(|member| {
                    find_im2_picture(&member.data, name).map(|found| member.data[found].to_vec())
                })
                .ok_or_else(|| fail(format!("{pack} has no {name} picture in {image}")))
        };
    let write = |name: &str, data: &[u8]| -> CoreResult<()> {
        let target = out.join(name);
        if paths.contains(name) || (target.is_file() && fs::read(&target)? == data) {
            return Ok(());
        }
        write_file(&target, data)
    };
    for language in 0..=6 {
        let pack = if language == 0 {
            SYSTEM_PACK.to_owned()
        } else {
            format!("gedit/system/esys_{language}.pak")
        };
        if !out.join(&pack).is_file() {
            continue;
        }
        let members = read_pack(&fs::read(out.join(&pack))?)?;
        let bank = write_im2(&[
            ("pause", picture(&members, "pause.img", "pause", &pack)?),
            (
                "whatsday",
                picture(&members, "whatsday.img", "whatsday", &pack)?,
            ),
        ])?;
        if language == 0 {
            write("gedit/system/sys.img", &bank)?;
        } else {
            write(&format!("gedit/system/sys_{language}.img"), &bank)?;
            continue;
        }
        let moved = [
            "gaiji.img",
            "syst04.img",
            "fuki256.img",
            "system.img",
            "pause.img",
            "whatsday.img",
        ];
        let mut common: Vec<Member> = members
            .iter()
            .filter(|member| {
                !moved
                    .iter()
                    .any(|name| member.name.eq_ignore_ascii_case(name.as_bytes()))
            })
            .cloned()
            .collect();
        common.push(Member {
            name: b"sys_cmn.img".to_vec(),
            data: write_im2(&[
                ("syst08", picture(&members, "system.img", "syst08", &pack)?),
                (
                    "pnplate",
                    picture(&members, "system.img", "pnplate", &pack)?,
                ),
                (
                    "dayclock",
                    picture(&members, "system.img", "dayclock", &pack)?,
                ),
            ])?,
        });
        write(PAL_COMMON_SYSTEM_PACK, &write_pack(&common)?)?;
    }
    Ok(())
}

/// PAL moved the weapon status sheet out of btlmenu.img into btlmenu2.img, and its battle menu
/// loads both, so the sheet has to leave btlmenu.img as well: entered twice, it runs the menu's
/// block past the fixed textures in VRAM. Returns whether the pack gained btlmenu2.img.
fn split_battle_menu(members: &mut Vec<Member>, path: &str) -> CoreResult<bool> {
    if find_member(members, "btlmenu2.img").is_some() {
        return Ok(false);
    }
    let Some(image) = members
        .iter_mut()
        .find(|member| member.name.eq_ignore_ascii_case(b"btlmenu.img"))
    else {
        return Ok(false);
    };
    let data = image.data.clone();
    let invalid = || fail(format!("invalid battle menu image in {path}"));
    if data.len() < 16 || &data[..3] != b"IM2" {
        return Err(invalid());
    }
    let count = le32(&data[4..])? as usize;
    if 16 + count * 48 > data.len() {
        return Err(invalid());
    }
    let build = |status: bool| -> CoreResult<(usize, Vec<u8>)> {
        let picked: Vec<usize> = (0..count)
            .filter(|&index| {
                (&data[16 + index * 48..16 + index * 48 + 10] == b"wepstatus\0") == status
            })
            .collect();
        let mut built = vec![0; 16 + picked.len() * 48];
        built[..16].copy_from_slice(&data[..16]);
        put32(&mut built, 4, picked.len() as u32);
        for (n, &index) in picked.iter().enumerate() {
            let entry = &data[16 + index * 48..16 + index * 48 + 48];
            let offset = le32(&entry[32..])? as usize;
            let end = if index + 1 < count {
                le32(&data[16 + (index + 1) * 48 + 32..])? as usize
            } else {
                data.len()
            };
            if offset < 16 + count * 48 || end > data.len() || end <= offset {
                return Err(invalid());
            }
            built[16 + n * 48..16 + n * 48 + 48].copy_from_slice(entry);
            let start = built.len() as u32;
            put32(&mut built, 16 + n * 48 + 32, start);
            built.extend_from_slice(&data[offset..end]);
        }
        Ok((picked.len(), built))
    };
    let (statuses, single) = build(true)?;
    if statuses != 1 {
        return Ok(false);
    }
    image.data = build(false)?.1;
    members.push(Member {
        name: b"btlmenu2.img".to_vec(),
        data: single,
    });
    Ok(true)
}

fn normalize(out: &Path, records: &[Record]) -> CoreResult<()> {
    normalize_icon_sheets(out, records)?;
    let paths: HashSet<&str> = records.iter().map(|record| record.path.as_str()).collect();
    if !is_ntsc_layout(&paths) {
        return Ok(());
    }
    let alias = |name: &str, source: &str| -> CoreResult<()> {
        let target = out.join(name);
        let original = out.join(source);
        if !paths.contains(name)
            && original.is_file()
            && !is_current(&target, fs::metadata(&original)?.len())
        {
            fs::create_dir_all(
                target
                    .parent()
                    .ok_or_else(|| fail("target has no parent"))?,
            )?;
            fs::copy(&original, &target)?;
        }
        Ok(())
    };
    for language in 1..=6 {
        alias(
            &format!("meswin/mes_tex_{language}.pak"),
            "meswin/mes_tex.pak",
        )?;
        alias(
            &format!("dun/pack/teximg2_{language}.pac"),
            "dun/pack/teximg2.pac",
        )?;
    }
    for language in ["jp", "us_e", "fr", "gr", "it", "sp"] {
        for map in 0..=6 {
            alias(
                &format!("dun/img/{language}/dname0{map}.img"),
                &format!("dun/img/us/dname0{map}.img"),
            )?;
        }
    }
    alias("gedit/system/editsys_2.mes", "gedit/system/editsys.bin")?;
    alias("meswin/system_2.mes", "meswin/systeme.bin")?;
    alias("meswin/system14_2.mes", "meswin/system14_1.mes")?;
    alias("opdat/optext_2.mes", "opdat/usa/fconv.bin")?;
    alias("opdat/optext_6.mes", "opdat/usa/fconv.bin")?;
    alias("titledat/title_eu.pak", "titledat/title.pak")?;
    for language in ['e', 'f', 'g', 'i', 's'] {
        alias(
            &format!("titledat/title_{language}.pak"),
            "titledat/title.pak",
        )?;
    }
    split_system_packs(out, &paths)?;
    for map in 1..=7 {
        let event = out.join(format!("dun/script/d{map:02}/event.stb"));
        if !event.is_file() {
            continue;
        }
        let data = fs::read(&event)?;
        if data.len() < 24 || le32(&data[20..])? as usize >= data.len() {
            return Err(fail(format!("invalid event script in {}", event.display())));
        }
        let messages = &data[le32(&data[20..])? as usize..];
        for language in 1..=6 {
            let name = format!("dun/script/d{map:02}/d{map:02}_{language}.mes");
            if !paths.contains(name.as_str()) {
                write_file(&out.join(&name), messages)?;
            }
        }
    }
    for record in records {
        let path = record.path.as_str();
        let rooms = path.starts_with("rmdat/rmdat") && path.ends_with(".pak");
        let pause = matches!(
            path,
            "opdat/dungeon/dungeon.pim"
                | "opdat/norn/norn.pak"
                | "opdat/norn2/norn2.pim"
                | "opdat/toan/toan.pim"
        );
        let battle = path.contains("/dungeon/dunmenu5.pak");
        if !rooms && !pause && !battle {
            continue;
        }
        let mut members = read_pack(&fs::read(out.join(path))?)?;
        let mut changed = false;
        for language in ['f', 'g', 'i', 's'] {
            changed |= if rooms {
                add_pack_alias(&mut members, &format!("start_{language}.img"), "start.img")
            } else {
                add_pack_alias(
                    &mut members,
                    &format!("pause_{language}.img"),
                    "pause_e.img",
                )
            };
        }
        if battle {
            changed |= split_battle_menu(&mut members, path)?;
        }
        if changed {
            write_file(&out.join("normalized").join(path), &write_pack(&members)?)?;
        }
    }
    Ok(())
}

/// Where texel (x, y) of an 8-bit TIM2 picture `width` texels wide lies in its pixel data: the
/// game's files keep them in the GS's PSMT8 block order.
fn tim2_t8_texel(x: usize, y: usize, width: usize) -> usize {
    (y & !0xF) * width
        + (x & !0xF) * 2
        + ((((y & !3) >> 1) + (y & 1)) & 7) * width * 2
        + ((x + (((y + 2) >> 2) & 1) * 4) & 7) * 4
        + ((y >> 1) & 1)
        + ((x >> 2) & 2)
}

/// The CLUT entry a pixel value names: the GS reads 8-bit CLUTs in CSM1 order, bits 3 and 4
/// swapped.
fn tim2_t8_slot(value: usize) -> usize {
    (value & !0x18) | ((value & 8) << 1) | ((value & 0x10) >> 1)
}

/// An itempack.img's 256 by 192 eight-bit itempack picture: its bank and where its pixels and
/// palette lie.
struct Sheet {
    bank: Vec<u8>,
    pixels: usize,
    clut: usize,
}

fn open_sheet(out: &Path, path: &str, prefer_normalized: bool) -> CoreResult<Option<Sheet>> {
    let normalized = out.join("normalized").join(path);
    let file = if prefer_normalized && normalized.is_file() {
        normalized
    } else {
        out.join(path)
    };
    if !file.is_file() {
        return Ok(None);
    }
    let bank = fs::read(&file)?;
    let Some(picture) = find_im2_picture(&bank, "itempack") else {
        return Ok(None);
    };
    let header = &bank[picture.start + 16..picture.end];
    if header.len() < 0x18
        || header[0x13] != 5
        || le16(header, 0x14) != 256
        || le16(header, 0x16) != 192
        || le16(header, 0x0E) != 256
    {
        return Ok(None);
    }
    let pixels = picture.start + 16 + le16(header, 0x0C);
    let clut = pixels + le32(&header[8..])? as usize;
    if pixels + 256 * 192 > bank.len() || clut + 1024 > bank.len() {
        return Ok(None);
    }
    Ok(Some(Sheet { bank, pixels, clut }))
}

/// The dungeon HUD's floor plates, in itempack.img's itempack picture: the plate the floor number
/// is drawn on and the back floors' over its right side. Every release's carry the word for floor,
/// the American and Japanese "Floor", and the HUD draws the number below it. The July 12 PAL
/// prototype left its British, French, German, Italian and Spanish ones blank (the retail release
/// lettered them), so they take American English's, each texel the nearest colour of the sheet's
/// own palette.
fn fill_floor_plates(out: &Path) -> CoreResult<()> {
    const PLATES: [(usize, usize, usize, usize); 2] =
        [(0x9A, 0x01, 0x66, 0x29), (0xDA, 0x2B, 0x26, 0x29)];
    let Some(american) = open_sheet(out, "commenu/a_usa/itempack.img", false)? else {
        return Ok(());
    };
    for language in ["a_eng", "a_fre", "a_ger", "a_ita", "a_spa"] {
        let path = format!("commenu/{language}/itempack.img");
        let Some(mut sheet) = open_sheet(out, &path, true)? else {
            continue;
        };
        for (left, top, width, height) in PLATES {
            for y in top..top + height {
                for x in left..left + width {
                    let value =
                        usize::from(american.bank[american.pixels + tim2_t8_texel(x, y, 256)]);
                    let at = american.clut + tim2_t8_slot(value) * 4;
                    let colour = &american.bank[at..at + 4];
                    let mut best = None;
                    for slot in 0..256 {
                        let entry = &sheet.bank[sheet.clut + slot * 4..sheet.clut + slot * 4 + 4];
                        if (colour[3] == 0) != (entry[3] == 0) {
                            continue;
                        }
                        let distance = if colour[3] == 0 {
                            0
                        } else {
                            entry[0].abs_diff(colour[0]) as u32
                                + entry[1].abs_diff(colour[1]) as u32
                                + entry[2].abs_diff(colour[2]) as u32
                                + 2 * entry[3].abs_diff(colour[3]) as u32
                        };
                        if best.is_none_or(|(_, nearest)| distance < nearest) {
                            best = Some((slot, distance));
                        }
                    }
                    if let Some((slot, _)) = best {
                        sheet.bank[sheet.pixels + tim2_t8_texel(x, y, 256)] =
                            tim2_t8_slot(slot) as u8;
                    }
                }
            }
        }
        let target = out.join("normalized").join(&path);
        if !target.is_file() || fs::read(&target)? != sheet.bank {
            write_file(&target, &sheet.bank)?;
        }
    }
    Ok(())
}

/// What the extraction can run, written to the data's root for the port: the release it came
/// from and the languages its disc carries, as the port's LanguageCode numbers.
const LANGUAGES_FILE: &str = "languages.json";

/// NTSC shipped American English alone. The disc's other languages are early drafts whose images
/// were never translated (its French and German title cards read "Nolun Village" and "Sun/Moon
/// Temple"), its British-named files have no town dialogue, and its Spanish is missing files.
const NTSC_LANGUAGES: &[u32] = &[1];

/// PAL: the five languages its select offers, English being British.
const PAL_LANGUAGES: &[u32] = &[2, 3, 4, 5, 6];

/// A release, known by a fingerprint of its DATA.HD2, the index of every file on the disc, which
/// differs from release to release.
#[derive(Clone, Copy)]
struct Release {
    name: &'static str,
    /// The index's size and FNV-1a hash.
    index: (u64, u64),
    languages: &'static [u32],
    /// Whether its other languages' floor plates are blank, to be filled from American English's.
    blank_floor_plates: bool,
}

const RELEASES: [Release; 2] = [
    Release {
        name: "NTSC 1.02",
        index: (294080, 0x806b_b1d4_3be8_a8dc),
        languages: NTSC_LANGUAGES,
        blank_floor_plates: false,
    },
    Release {
        name: "PAL prototype (July 12, 2001)",
        index: (312864, 0x5942_a325_63c2_a8b1),
        languages: PAL_LANGUAGES,
        blank_floor_plates: true,
    },
];

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The release by its index, or for one the table lacks, its family by the file layout: NTSC's
/// languages where the disc has NTSC's system pack alone, PAL's otherwise.
fn release_of(hd2: &[u8], records: &[Record]) -> Release {
    let index = (hd2.len() as u64, fnv1a64(hd2));
    if let Some(known) = RELEASES.iter().find(|release| release.index == index) {
        return *known;
    }
    let paths: HashSet<&str> = records.iter().map(|record| record.path.as_str()).collect();
    let (name, languages) = if is_ntsc_layout(&paths) {
        ("unknown release, NTSC layout", NTSC_LANGUAGES)
    } else {
        ("unknown release, PAL layout", PAL_LANGUAGES)
    };
    Release {
        name,
        index,
        languages,
        blank_floor_plates: false,
    }
}

fn write_languages(out: &Path, release: &Release) -> CoreResult<()> {
    let languages: Vec<String> = release.languages.iter().map(u32::to_string).collect();
    let text = format!(
        "{{\"release\": \"{}\", \"languages\": [{}]}}\n",
        release.name,
        languages.join(", ")
    );
    write_file(&out.join(LANGUAGES_FILE), text.as_bytes())
}

#[derive(Debug, Default)]
pub struct Report {
    /// Files the game can reach, and their bytes.
    pub files: u64,
    pub bytes: u64,
    /// Of those, the ones copied and the ones already there byte for byte.
    pub written: u64,
    pub kept: u64,
    /// Records naming a file listed earlier, which the game never finds.
    pub duplicates: u64,
    /// Records whose sector count is short of their size.
    pub warnings: u64,
    /// Records of files named outside ASCII, which the game never loads and are left out.
    pub omitted: u64,
    pub release: &'static str,
}

pub fn extract(source: &Path, out: &Path, jobs: usize) -> CoreResult<Report> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(jobs.clamp(1, 256))
        .build()
        .map_err(|err| fail(format!("cannot start extraction workers: {err}")))?
        .install(|| extract_with_progress(source, out, jobs, &Phase::noop()))
}

/// Extracts every reachable file in parallel, reusing files already there byte for byte, then
/// writes the index's copy, the releases' fixes and `languages.json` in order.
pub fn extract_with_progress(
    source: &Path,
    out: &Path,
    jobs: usize,
    phase: &Phase,
) -> CoreResult<Report> {
    if jobs == 0 || jobs > 256 {
        return Err(fail("jobs must be in 1..=256"));
    }
    let disc = open_disc(source)?;
    let hd2 = disc.hd2.read()?;
    let records = parse_index(&hd2)?;
    let warnings = check_fits(&records, disc.dat.size, phase)?;
    let (records, omitted) = loadable(records);
    if omitted > 0 {
        phase.log(format!(
            "leaving out {omitted} files named outside ASCII, which the game never loads"
        ));
    }
    let (files, duplicates) = reachable(&records, phase);
    let bytes: u64 = files.iter().map(|record| u64::from(record.size)).sum();
    phase.log(format!(
        "extracting {} files ({}) with {jobs} workers",
        files.len(),
        super::progress::size(bytes)
    ));
    phase.set_total(bytes);
    fs::create_dir_all(out)?;
    let kept = AtomicU64::new(0);
    files.par_iter().try_for_each(|record| -> CoreResult<()> {
        let extent = Extent {
            file: disc.dat.file.clone(),
            offset: disc.dat.offset + u64::from(record.sector) * SECTOR,
            size: u64::from(record.size),
        };
        let target = out.join(&record.path);
        if extent.size >= 8 << 20 {
            phase.item(&record.path);
        }
        if extent_matches(&extent, &target)
            .map_err(|err| fail(format!("{}: resume comparison: {}", record.path, err.0)))?
        {
            kept.fetch_add(1, Ordering::Relaxed);
        } else {
            copy_extent(&extent, &target)
                .map_err(|err| fail(format!("{}: {}", record.path, err.0)))?;
        }
        phase.inc(extent.size);
        phase.add_bytes(extent.size);
        Ok(())
    })?;
    write_file(&out.join("data.hd2"), &hd2)?;
    normalize(out, &records)?;
    let release = release_of(&hd2, &records);
    phase.log(format!("release: {}", release.name));
    if release.blank_floor_plates {
        fill_floor_plates(out)?;
    }
    write_languages(out, &release)?;
    let kept = kept.into_inner();
    Ok(Report {
        files: files.len() as u64,
        bytes,
        written: files.len() as u64 - kept,
        kept,
        duplicates,
        warnings,
        omitted,
        release: release.name,
    })
}

pub fn list_text(source: &Path) -> CoreResult<String> {
    let disc = open_disc(source)?;
    let records = parse_index(&disc.hd2.read()?)?;
    let (loadable, omitted) = loadable(records.clone());
    let (files, duplicates) = reachable(&loadable, &Phase::noop());
    let mut output = format!(
        "source: {} ({})\nDATA.DAT: {} bytes, DATA.HD2: {} bytes\n",
        disc.source.display(),
        if disc.image {
            "ISO 9660 image"
        } else {
            "directory"
        },
        disc.dat.size,
        disc.hd2.size
    );
    for record in &records {
        output.push_str(&format!(
            "{}\t{}\t{}\n",
            record.display(),
            record.size,
            record.sector
        ));
    }
    output.push_str(&format!(
        "{} entries, {} files, {duplicates} duplicates, {omitted} left out, {} bytes\n",
        records.len(),
        files.len(),
        files
            .iter()
            .map(|record| u64::from(record.size))
            .sum::<u64>()
    ));
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(size: usize, seed: usize) -> Vec<u8> {
        (0..size)
            .map(|i| ((i * 31 + seed * 17 + 1) % 251 + 1) as u8)
            .collect()
    }

    fn named(name: &str, data: Vec<u8>) -> Member {
        Member {
            name: name.as_bytes().to_vec(),
            data,
        }
    }

    /// A pack as the disc's: 76-byte entries, then each member's data on 16 bytes.
    fn make_pack(members: &[Member]) -> Vec<u8> {
        let mut pack = vec![0; 76 * (members.len() + 1)];
        for (index, member) in members.iter().enumerate() {
            let entry = 76 * index;
            let data = pack.len().next_multiple_of(16);
            pack.resize(data, 0);
            pack.extend_from_slice(&member.data);
            pack[entry..entry + member.name.len()].copy_from_slice(&member.name);
            put32(&mut pack, entry + 64, (data - entry) as u32);
            put32(&mut pack, entry + 68, member.data.len() as u32);
            put32(&mut pack, entry + 72, 76);
        }
        pack
    }

    /// DATA.DAT and DATA.HD2 as the disc lays them out: each file on its own sector, one empty
    /// sector in, the tail of its last sector filled so a reader copying whole sectors is caught,
    /// and the index's closing all-zero record.
    fn make_disc(files: &[(&str, Vec<u8>)]) -> (Vec<u8>, Vec<u8>) {
        let mut hd2 = vec![0; 32 * (files.len() + 1)];
        let mut dat = vec![0xCD; SECTOR as usize];
        for (index, (name, data)) in files.iter().enumerate() {
            let sector = dat.len() / SECTOR as usize;
            let name_offset = hd2.len() as u32;
            hd2.extend_from_slice(name.as_bytes());
            hd2.push(0);
            dat.extend_from_slice(data);
            dat.resize(dat.len().next_multiple_of(SECTOR as usize), 0xCD);
            put32(&mut hd2, 32 * index, name_offset);
            put32(&mut hd2, 32 * index + 4, 0xDEAD_BEEF);
            put32(&mut hd2, 32 * index + 16, (sector * SECTOR as usize) as u32);
            put32(&mut hd2, 32 * index + 20, data.len() as u32);
            put32(&mut hd2, 32 * index + 24, sector as u32);
            put32(
                &mut hd2,
                32 * index + 28,
                sectors_for(data.len() as u64) as u32,
            );
        }
        let end = hd2.len() as u32 - 1;
        put32(&mut hd2, 32 * files.len(), end);
        (dat, hd2)
    }

    fn write_disc(root: &Path, files: &[(&str, Vec<u8>)]) -> CoreResult<(Vec<u8>, Vec<u8>)> {
        let (dat, hd2) = make_disc(files);
        fs::create_dir_all(root)?;
        fs::write(root.join("DATA.DAT"), &dat)?;
        fs::write(root.join("data.HD2"), &hd2)?;
        Ok((dat, hd2))
    }

    fn iso_record(name: &[u8], sector: u32, size: u32, directory: bool) -> Vec<u8> {
        let length = 33 + name.len() + usize::from(name.len().is_multiple_of(2));
        let mut bytes = vec![0; length];
        bytes[0] = length as u8;
        bytes[2..6].copy_from_slice(&sector.to_le_bytes());
        bytes[6..10].copy_from_slice(&sector.to_be_bytes());
        bytes[10..14].copy_from_slice(&size.to_le_bytes());
        bytes[14..18].copy_from_slice(&size.to_be_bytes());
        bytes[25] = if directory { 2 } else { 0 };
        bytes[28] = 1;
        bytes[31] = 1;
        bytes[32] = name.len() as u8;
        bytes[33..33 + name.len()].copy_from_slice(name);
        bytes
    }

    /// An ISO 9660 image of the pair, its root directory at sector 18 and the files after it.
    fn make_iso(dat: &[u8], hd2: &[u8]) -> Vec<u8> {
        let sector = SECTOR as usize;
        let hd2_sector = 19 + dat.len().div_ceil(sector);
        let mut iso = vec![0; (hd2_sector + hd2.len().div_ceil(sector)) * sector];
        let descriptor = 16 * sector;
        iso[descriptor] = 1;
        iso[descriptor + 1..descriptor + 6].copy_from_slice(b"CD001");
        iso[descriptor + 6] = 1;
        iso[descriptor + 128..descriptor + 130].copy_from_slice(&(SECTOR as u16).to_le_bytes());
        iso[descriptor + 130..descriptor + 132].copy_from_slice(&(SECTOR as u16).to_be_bytes());
        let root = iso_record(&[0], 18, SECTOR as u32, true);
        iso[descriptor + 156..descriptor + 156 + root.len()].copy_from_slice(&root);
        let terminator = 17 * sector;
        iso[terminator] = 255;
        iso[terminator + 1..terminator + 6].copy_from_slice(b"CD001");
        iso[terminator + 6] = 1;
        let mut entries = root;
        entries.extend(iso_record(&[1], 18, SECTOR as u32, true));
        entries.extend(iso_record(b"DATA.DAT;1", 19, dat.len() as u32, false));
        entries.extend(iso_record(
            b"Data.Hd2;1",
            hd2_sector as u32,
            hd2.len() as u32,
            false,
        ));
        iso[18 * sector..18 * sector + entries.len()].copy_from_slice(&entries);
        iso[19 * sector..19 * sector + dat.len()].copy_from_slice(dat);
        iso[hd2_sector * sector..hd2_sector * sector + hd2.len()].copy_from_slice(hd2);
        iso
    }

    /// The game's own paths, as DATA.HD2 spells them: backslashes, mixed case, one with a leading
    /// separator, a later duplicate of the first, sizes on both sides of a sector, an empty file,
    /// and a map folder's label named outside ASCII, as the developers' are.
    fn standard_files() -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("DUN\\PACK\\MAINDAT.PAC", pattern(5000, 10)),
            ("img\\Title.img", pattern(2048, 11)),
            ("\\sound\\bgm\\b01.SND", pattern(1, 12)),
            (
                "rmdat\\rmdat1.pak",
                make_pack(&[
                    named("info.cfg", pattern(37, 1)),
                    named("MODEL.MDS", pattern(300, 2)),
                ]),
            ),
            ("meswin\\systeme.bin", pattern(2047, 13)),
            ("dun\\pack\\maindat.pac", pattern(100, 14)),
            ("empty.bin", Vec::new()),
            ("gedit\\s48\\48ドラン戦", Vec::new()),
        ]
    }

    fn check_extracted(out: &Path, files: &[(&str, Vec<u8>)], hd2: &[u8]) -> CoreResult<()> {
        for (path, index) in [
            ("dun/pack/maindat.pac", 0),
            ("img/title.img", 1),
            ("sound/bgm/b01.snd", 2),
            ("rmdat/rmdat1.pak", 3),
            ("meswin/systeme.bin", 4),
            ("empty.bin", 6),
        ] {
            assert_eq!(fs::read(out.join(path))?, files[index].1, "{path}");
        }
        assert_eq!(fs::read(out.join("data.hd2"))?, hd2);
        // Listed rather than looked up: macOS's file system ignores case.
        assert!(fs::read_dir(out)?.all(|entry| entry.unwrap().file_name() != "DUN"));
        Ok(())
    }

    #[test]
    fn index_parses_as_pal() -> CoreResult<()> {
        let (_, hd2) = make_disc(&standard_files());
        let records = parse_index(&hd2)?;
        assert_eq!(records.len(), 8);
        assert_eq!(records[0].name, b"DUN\\PACK\\MAINDAT.PAC");
        assert_eq!(records[0].path, "dun/pack/maindat.pac");
        assert_eq!(
            (records[0].size, records[0].sector, records[0].sectors),
            (5000, 1, 3)
        );
        assert_eq!(records[2].path, "sound/bgm/b01.snd");
        assert_eq!((records[6].size, records[6].sectors), (0, 0));
        Ok(())
    }

    #[test]
    fn index_rejects_bad_records() {
        let (_, hd2) = make_disc(&standard_files());
        let rejects = |index: &[u8], needle: &str| {
            let error = parse_index(index).err().expect("rejected").0;
            assert!(error.contains(needle), "{error}");
        };
        rejects(&hd2[..31], "too short");
        let mut overcount = hd2.clone();
        put32(&mut overcount, 0, hd2.len() as u32 + 32);
        rejects(&overcount, "claims");
        let mut far_name = hd2.clone();
        put32(&mut far_name, 32, hd2.len() as u32);
        rejects(&far_name, "record 1: name offset");
        rejects(&hd2[..hd2.len() - 1], "unterminated");
        rejects(
            &make_disc(&[("dun\\..\\..\\etc\\passwd", pattern(4, 0))]).1,
            "unusable path",
        );
        let mut negative = hd2.clone();
        put32(&mut negative, 32 + 20, 0xFFFF_FFF0);
        rejects(&negative, "negative");
    }

    #[test]
    fn iso_and_directory_extract_alike() -> CoreResult<()> {
        let temp = tempfile::tempdir()?;
        let files = standard_files();
        let (dat, hd2) = write_disc(&temp.path().join("disc"), &files)?;
        fs::write(temp.path().join("disc/SYSTEM.CNF"), pattern(10, 0))?;
        let iso = temp.path().join("dark cloud.iso");
        fs::write(&iso, make_iso(&dat, &hd2))?;
        for (source, out) in [
            (temp.path().join("disc"), "from-directory"),
            (iso, "from-iso"),
        ] {
            let out = temp.path().join(out);
            let report = extract(&source, &out, 2)?;
            assert_eq!(
                (
                    report.files,
                    report.written,
                    report.duplicates,
                    report.warnings,
                    report.omitted
                ),
                (6, 6, 1, 0, 1)
            );
            assert!(!out.join("gedit/s48").exists());
            check_extracted(&out, &files, &hd2)?;
            // A release the table lacks, without NTSC's layout: PAL's five languages.
            assert_eq!(
                fs::read_to_string(out.join("languages.json"))?,
                "{\"release\": \"unknown release, PAL layout\", \"languages\": [2, 3, 4, 5, 6]}\n"
            );
        }
        assert!(
            list_text(&temp.path().join("disc"))?
                .contains("8 entries, 6 files, 1 duplicates, 1 left out")
        );
        assert!(open_disc(&temp.path().join("from-iso")).is_err());
        Ok(())
    }

    #[test]
    fn extraction_reuses_only_exact_files() -> CoreResult<()> {
        let temp = tempfile::tempdir()?;
        let files = standard_files();
        let (_, hd2) = write_disc(&temp.path().join("disc"), &files)?;
        let out = temp.path().join("data");
        extract(&temp.path().join("disc"), &out, 2)?;
        let again = extract(&temp.path().join("disc"), &out, 2)?;
        assert_eq!((again.written, again.kept), (0, 6));
        fs::write(out.join("img/title.img"), pattern(2048, 99))?;
        let repaired = extract(&temp.path().join("disc"), &out, 2)?;
        assert_eq!((repaired.written, repaired.kept), (1, 5));
        check_extracted(&out, &files, &hd2)
    }

    #[test]
    fn truncated_discs_are_rejected_before_writing() -> CoreResult<()> {
        let temp = tempfile::tempdir()?;
        let (dat, hd2) = make_disc(&standard_files());
        fs::create_dir_all(temp.path().join("cut"))?;
        fs::write(
            temp.path().join("cut/DATA.DAT"),
            &dat[..dat.len() - 2 * SECTOR as usize],
        )?;
        fs::write(temp.path().join("cut/DATA.HD2"), &hd2)?;
        let error = extract(&temp.path().join("cut"), &temp.path().join("data"), 2)
            .expect_err("rejected")
            .0;
        assert!(error.contains("DATA.DAT is truncated"), "{error}");
        assert!(!temp.path().join("data").exists());

        let mut image = make_iso(&dat, &hd2);
        image.truncate(image.len() - SECTOR as usize);
        fs::write(temp.path().join("cut.iso"), image)?;
        assert!(open_disc(&temp.path().join("cut.iso")).is_err());
        fs::write(temp.path().join("tiny.iso"), [0u8; 100])?;
        assert!(open_disc(&temp.path().join("tiny.iso")).is_err());
        Ok(())
    }

    #[test]
    fn releases_are_known_by_their_index() {
        // FNV-1a's published value for "a".
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        for release in &RELEASES {
            assert!(!release.languages.is_empty());
            assert_eq!(
                RELEASES
                    .iter()
                    .filter(|other| other.index == release.index)
                    .count(),
                1
            );
        }
        assert_eq!(
            release_of(&pattern(64, 3), &[]).name,
            "unknown release, PAL layout"
        );
    }

    /// An IM2 bank of 16 by 16 eight-bit pictures, each with its own palette and pixel fill.
    fn make_icon_bank(pictures: &[(&str, u32, u8)]) -> Vec<u8> {
        let mut bank = vec![0; 16 + pictures.len() * 48];
        bank[..3].copy_from_slice(b"IM2");
        put32(&mut bank, 4, pictures.len() as u32);
        for (index, &(name, colour, pixel)) in pictures.iter().enumerate() {
            bank[16 + index * 48..16 + index * 48 + name.len()].copy_from_slice(name.as_bytes());
            let offset = bank.len() as u32;
            put32(&mut bank, 16 + index * 48 + 32, offset);
            let mut tim = vec![0; 16 + 48 + 256 + 1024];
            tim[..4].copy_from_slice(b"TIM2");
            put32(&mut tim, 16, 0x30030); // the wrong total the game's images carry
            put32(&mut tim, 16 + 4, 1024);
            put32(&mut tim, 16 + 8, 256);
            tim[16 + 12] = 0x30;
            tim[16 + 15] = 0x01;
            tim[16 + 19] = 5;
            tim[16 + 20] = 16;
            tim[16 + 22] = 16;
            tim[64..64 + 256].fill(pixel);
            for c in 0..256 {
                put32(&mut tim, 64 + 256 + c * 4, colour + c as u32);
            }
            bank.extend(tim);
        }
        bank
    }

    fn picture(bank: &[u8], name: &str) -> Vec<u8> {
        bank[find_im2_picture(bank, name).expect(name)].to_vec()
    }

    fn picture_names(bank: &[u8]) -> Vec<String> {
        (0..le32(&bank[4..]).unwrap() as usize)
            .map(|index| {
                let entry = &bank[16 + index * 48..16 + index * 48 + 32];
                String::from_utf8_lossy(&entry[..entry.iter().position(|&b| b == 0).unwrap()])
                    .into_owned()
            })
            .collect()
    }

    /// An NTSC town system pack: its pause and day-of-adventure pictures of the given colour.
    fn ntsc_system_pack(colour: u32) -> Vec<u8> {
        make_pack(&[
            named(
                "whatsday.img",
                make_icon_bank(&[("whatsday", colour + 1, 1)]),
            ),
            named("cursor.img", pattern(8, 1)),
            named("gaiji.img", pattern(8, 2)),
            named(
                "system.img",
                make_icon_bank(&[
                    ("dayclock", 0x10, 3),
                    ("syst08", 0x20, 4),
                    ("skip_bord", 0x30, 5),
                    ("pause", 0x40, 6),
                    ("pnplate", 0x50, 7),
                ]),
            ),
            named("takara.mds", pattern(8, 3)),
            named("pause.img", make_icon_bank(&[("pause", colour, 2)])),
        ])
    }

    #[test]
    fn ntsc_assets_take_pal_names() -> CoreResult<()> {
        let temp = tempfile::tempdir()?;
        let start = pattern(32, 7);
        let mut event = vec![0; 24];
        put32(&mut event, 20, 24);
        event.extend([1, 2, 3, 4]);
        let mut image = vec![0; 16 + 2 * 48 + 8];
        image[..3].copy_from_slice(b"IM2");
        put32(&mut image, 4, 2);
        image[16..21].copy_from_slice(b"other");
        put32(&mut image, 16 + 32, 16 + 2 * 48);
        image[64..73].copy_from_slice(b"wepstatus");
        put32(&mut image, 64 + 32, 16 + 2 * 48 + 4);
        for (index, byte) in image[16 + 2 * 48..].iter_mut().enumerate() {
            *byte = index as u8 + 1;
        }
        let files = vec![
            ("gedit/system/esys.pak", ntsc_system_pack(0x100)),
            ("meswin/mes_tex.pak", pattern(16, 2)),
            (
                "rmdat/rmdat1.pak",
                make_pack(&[named("start.img", start.clone())]),
            ),
            ("dun/img/us/dname00.img", pattern(16, 3)),
            ("dun/script/d01/event.stb", event),
            (
                "commenu/a_eng/dungeon/dunmenu5.pak",
                make_pack(&[named("btlmenu.img", image.clone())]),
            ),
        ];
        write_disc(&temp.path().join("disc"), &files)?;
        let out = temp.path().join("data");
        extract(&temp.path().join("disc"), &out, 2)?;

        assert_eq!(fs::read(out.join("meswin/mes_tex_2.pak"))?, files[1].1);
        assert_eq!(fs::read(out.join("dun/img/us_e/dname00.img"))?, files[3].1);
        let rooms = read_pack(&fs::read(out.join("normalized/rmdat/rmdat1.pak"))?)?;
        assert_eq!(rooms.len(), 5);
        assert_eq!(
            (rooms[1].name.as_slice(), &rooms[1].data),
            (&b"start_f.img"[..], &start)
        );
        assert_eq!(
            fs::read(out.join("dun/script/d01/d01_1.mes"))?,
            [1, 2, 3, 4]
        );
        assert_eq!(
            fs::read(out.join("dun/script/d01/d01_2.mes"))?,
            [1, 2, 3, 4]
        );
        assert_eq!(
            fs::read_to_string(out.join("languages.json"))?,
            "{\"release\": \"unknown release, NTSC layout\", \"languages\": [1]}\n"
        );
        let battle = read_pack(&fs::read(
            out.join("normalized/commenu/a_eng/dungeon/dunmenu5.pak"),
        )?)?;
        assert_eq!(battle.len(), 2);
        assert_eq!(battle[1].name, b"btlmenu2.img");
        assert_eq!(le32(&battle[1].data[4..])?, 1);
        assert_eq!(battle[1].data[64..], image[116..]);
        assert_eq!(le32(&battle[0].data[4..])?, 1);
        assert_eq!(&battle[0].data[16..22], b"other\0");
        assert_eq!(le32(&battle[0].data[16 + 32..])?, 64);
        assert_eq!(battle[0].data[64..], image[112..116]);

        let again = extract(&temp.path().join("disc"), &out, 2)?;
        assert_eq!((again.written, again.kept), (0, 6));
        Ok(())
    }

    #[test]
    fn icon_sheets_take_itempacks_palette() -> CoreResult<()> {
        let temp = tempfile::tempdir()?;
        let american = make_icon_bank(&[("quickchara", 0x300, 3), ("wepicon", 0x200, 4)]);
        let english = make_icon_bank(&[("wepicon", 0x100, 5)]);
        let files = vec![
            (
                "commenu/a_eng/itempack.img",
                make_icon_bank(&[("itempack", 0x100, 1)]),
            ),
            (
                "commenu/a_eng/quickchr.pac",
                make_pack(&[
                    named("qchr.mes", pattern(8, 2)),
                    named("quickchr.img", american.clone()),
                ]),
            ),
            (
                "commenu/a_eng/itemlst.img",
                make_icon_bank(&[("wepicon", 0x200, 4), ("itemicon", 0x200, 7)]),
            ),
            (
                "commenu/a_eng/dunenter/dunenter2.pak",
                make_pack(&[named("dunenter.img", english.clone())]),
            ),
            (
                "commenu/a_eng/kgetoan2.img",
                make_icon_bank(&[("wepicon", 0x200, 4)]),
            ),
            (
                "commenu/a_eng/_charatex.img",
                make_icon_bank(&[("wepicon", 0x200, 4)]),
            ),
            ("commenu/a_eng/manual/m10.pac", pattern(80, 9)),
            (
                "commenu/a_fre/itempack.img",
                make_icon_bank(&[("itempack", 0x200, 6)]),
            ),
            (
                "commenu/a_fre/quickchr.pac",
                make_pack(&[named("quickchr.img", american.clone())]),
            ),
            (
                "commenu/a_fre/dunenter/dunenter2.pak",
                make_pack(&[named("dunenter.img", english.clone())]),
            ),
        ];
        write_disc(&temp.path().join("disc"), &files)?;
        let out = temp.path().join("data");
        extract(&temp.path().join("disc"), &out, 2)?;

        // English: the American sheets take the English one; the rest of each file is as it was.
        let donor = picture(&english, "wepicon");
        let quick = read_pack(&fs::read(
            out.join("normalized/commenu/a_eng/quickchr.pac"),
        )?)?;
        assert_eq!(quick.len(), 2);
        assert_eq!(quick[0].data, pattern(8, 2));
        assert_eq!(picture(&quick[1].data, "wepicon"), donor);
        assert_eq!(
            picture(&quick[1].data, "quickchara"),
            picture(&american, "quickchara")
        );
        assert_eq!(
            picture(
                &fs::read(out.join("normalized/commenu/a_eng/itemlst.img"))?,
                "wepicon"
            ),
            donor
        );
        // The main menu's sheet too, but not a leftover the game never loads, and a .pac without
        // a pack.
        assert_eq!(
            picture(
                &fs::read(out.join("normalized/commenu/a_eng/kgetoan2.img"))?,
                "wepicon"
            ),
            donor
        );
        assert!(!out.join("normalized/commenu/a_eng/_charatex.img").exists());
        assert!(!out.join("normalized/commenu/a_eng/manual/m10.pac").exists());
        // French: its sheet already shares itempack's palette.
        assert!(!out.join("normalized/commenu/a_fre/quickchr.pac").exists());
        Ok(())
    }

    /// PAL's town system packs, out of NTSC's: the common pack without the pictures PAL moved, with
    /// sys_cmn.img three of system.img's pictures, and each language's pause and day-of-adventure
    /// pictures as sys_N.img.
    #[test]
    fn ntsc_system_packs_split_as_pals() -> CoreResult<()> {
        let temp = tempfile::tempdir()?;
        let files = vec![
            ("gedit/system/esys.pak", ntsc_system_pack(0x100)),
            ("gedit/system/esys_1.pak", ntsc_system_pack(0x200)),
        ];
        write_disc(&temp.path().join("disc"), &files)?;
        let out = temp.path().join("data");
        extract(&temp.path().join("disc"), &out, 2)?;

        let ntsc = read_pack(&files[0].1)?;
        let common = read_pack(&fs::read(out.join("gedit/system/esys_cmn.pak"))?)?;
        let names: Vec<&[u8]> = common.iter().map(|member| member.name.as_slice()).collect();
        assert_eq!(names, [&b"cursor.img"[..], b"takara.mds", b"sys_cmn.img"]);
        assert_eq!(
            picture_names(&common[2].data),
            ["syst08", "pnplate", "dayclock"]
        );
        for name in ["syst08", "pnplate", "dayclock"] {
            assert_eq!(
                picture(&common[2].data, name),
                picture(&ntsc[3].data, name),
                "{name}"
            );
        }
        assert_eq!(le32(&common[2].data[16 + 32..])?, 16 + 3 * 48);
        for (language, bank) in [(0, "gedit/system/sys.img"), (1, "gedit/system/sys_1.img")] {
            let pack = read_pack(&files[language].1)?;
            let bank = fs::read(out.join(bank))?;
            assert_eq!(picture_names(&bank), ["pause", "whatsday"]);
            assert_eq!(picture(&bank, "pause"), picture(&pack[5].data, "pause"));
            assert_eq!(
                picture(&bank, "whatsday"),
                picture(&pack[0].data, "whatsday")
            );
        }
        assert!(!out.join("gedit/system/sys_2.img").exists());
        Ok(())
    }

    /// An itempack.img: one 256 by 192 eight-bit picture named itempack, every texel `fill` but
    /// those of the floor plate's first row, `plate`, and colour i at the CLUT slot of value i.
    fn make_itempack(colours: &[u32], fill: u8, plate: u8) -> Vec<u8> {
        let pixels = 256 * 192;
        let mut tim = vec![0; 16 + 48 + pixels + 1024];
        tim[..4].copy_from_slice(b"TIM2");
        put32(&mut tim, 16 + 4, 1024);
        put32(&mut tim, 16 + 8, pixels as u32);
        tim[16 + 0x0C] = 0x30;
        tim[16 + 0x0F] = 0x01;
        tim[16 + 0x13] = 5;
        tim[16 + 0x15] = 1;
        tim[16 + 0x16] = 192;
        tim[64..64 + pixels].fill(fill);
        for x in 0x9A..0x9A + 0x66 {
            tim[64 + tim2_t8_texel(x, 1, 256)] = plate;
        }
        for (value, &colour) in colours.iter().enumerate() {
            put32(&mut tim, 64 + pixels + tim2_t8_slot(value) * 4, colour);
        }
        let mut bank = vec![0; 16 + 48];
        bank[..3].copy_from_slice(b"IM2");
        put32(&mut bank, 4, 1);
        bank[16..24].copy_from_slice(b"itempack");
        put32(&mut bank, 16 + 32, 16 + 48);
        bank.extend(tim);
        bank
    }

    /// The July 12 PAL prototype's blank floor plates take American English's, in each sheet's
    /// palette.
    #[test]
    fn blank_floor_plates_take_american_englishs() -> CoreResult<()> {
        let temp = tempfile::tempdir()?;
        let mut american: Vec<u32> = (0..256).map(|i| 0x8000_0000 | (i * 0x01_0101)).collect();
        let mut british: Vec<u32> = (0..256)
            .map(|i| 0x8000_0000 | ((255 - i) * 0x01_0101))
            .collect();
        american[0] = 0; // transparent
        british[7] = 0x0012_3456; // transparent, elsewhere
        let english = make_itempack(&british, 200, 200);
        write_file(
            &temp.path().join("commenu/a_usa/itempack.img"),
            &make_itempack(&american, 0, 40),
        )?;
        write_file(&temp.path().join("commenu/a_eng/itempack.img"), &english)?;
        fill_floor_plates(temp.path())?;

        let sheet = fs::read(temp.path().join("normalized/commenu/a_eng/itempack.img"))?;
        let value = |x, y| sheet[16 + 48 + 64 + tim2_t8_texel(x, y, 256)];
        assert_eq!(value(0x9A, 1), 255 - 40); // grey 40, at its place in the British palette
        assert_eq!(value(0x9A + 0x65, 1), 255 - 40);
        assert_eq!(value(0x9A, 2), 7); // the American plate's transparent texels
        assert_eq!(value(0xDA, 0x2B + 0x28), 7); // and the back floor's plate
        assert_eq!(value(0x99, 1), 200); // outside the plates: the sheet's own
        assert_eq!(value(0x9A, 0x2A), 200);
        assert_eq!(
            fs::read(temp.path().join("commenu/a_eng/itempack.img"))?,
            english
        );
        assert!(
            !temp
                .path()
                .join("normalized/commenu/a_usa/itempack.img")
                .exists()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires KAZAROV_HD2_ISO pointing at a local Dark Cloud disc image"]
    fn retail_disc_extracts_every_reachable_file() -> CoreResult<()> {
        let path = std::env::var_os("KAZAROV_HD2_ISO")
            .ok_or_else(|| fail("KAZAROV_HD2_ISO is missing"))?;
        let temp = tempfile::tempdir()?;
        let report = extract(Path::new(&path), temp.path(), 8)?;
        let records = parse_index(&fs::read(temp.path().join("data.hd2"))?)?;
        assert_eq!(
            report.files + report.duplicates + report.omitted,
            records.len() as u64
        );
        for record in reachable(&loadable(records.clone()).0, &Phase::noop()).0 {
            assert!(
                is_current(&temp.path().join(&record.path), u64::from(record.size)),
                "{}",
                record.path
            );
        }
        assert!(temp.path().join("languages.json").is_file());
        Ok(())
    }
}
