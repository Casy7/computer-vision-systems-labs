-- Допоміжний модуль Lua для завантаження та форматування даних CSV у LaTeX-таблиці
local RES = "../results/"

local function parse_csv_line(line)
  local res = {}
  local pos = 1
  local len = #line
  while pos <= len do
    if line:sub(pos, pos) == '"' then
      local _, end_pos, val = line:find('^"(.-)"', pos)
      res[#res + 1] = val or ""
      pos = (end_pos or pos) + 2
    else
      local next_comma = line:find(",", pos, true)
      if next_comma then
        res[#res + 1] = line:sub(pos, next_comma - 1)
        pos = next_comma + 1
      else
        res[#res + 1] = line:sub(pos)
        break
      end
    end
  end
  return res
end

local function read_csv(name)
  local rows = {}
  local path = RES .. name
  local f = io.open(path, "r")
  if not f then
    f = io.open("results/" .. name, "r")
  end
  if not f then
    tex.print("\\textbf{ПОМИЛКА: " .. name .. " не знайдено}")
    return {}
  end
  for line in f:lines() do
    line = line:gsub("\r", "")
    if #line > 0 then
      rows[#rows + 1] = parse_csv_line(line)
    end
  end
  f:close()
  if #rows > 0 then
    table.remove(rows, 1)
  end
  return rows
end

local function out(cells)
  tex.print(table.concat(cells, " & ") .. " \\\\")
end

local function f(x, d)
  local v = tonumber(x)
  if not v then return tostring(x) end
  return string.format("%." .. (d or 2) .. "f", v)
end

local function esc_tex(s)
  return (tostring(s):gsub("&", "\\&"):gsub("_", "\\_"):gsub("%%", "\\%%"):gsub("#", "\\#"):gsub("<", "$<$"):gsub(">", "$>$"))
end

-- Таблиця 1: Адаптація внутрішніх параметрів камери (Частина A, Крок 1)
function tab_intrinsics()
  local rows = read_csv("a_intrinsics_adaptation.csv")
  for _, r in ipairs(rows) do
    out({
      esc_tex(r[1]),
      "\\texttt{" .. esc_tex(r[2]) .. "}",
      r[3],
      f(r[4], 2) .. "~пк",
      f(r[5], 2) .. "~пк",
      f(r[6], 2) .. "~пк",
      f(r[7], 2) .. "~пк"
    })
  end
end

-- Таблиця 2: Порівняння моделей взаємного орієнтування стереопари (Частина A, Крок 3)
function tab_stereo_models()
  local rows = read_csv("a_stereo_models.csv")
  for _, r in ipairs(rows) do
    out({
      esc_tex(r[1]),
      f(r[2], 3) .. "$^\\circ$",
      "(" .. f(r[3], 2) .. "; " .. f(r[4], 2) .. "; " .. f(r[5], 2) .. ")",
      f(r[6], 2) .. "~пк",
      f(r[7], 3) .. "~пк",
      f(r[8], 3) .. "~пк",
      f(r[9], 1) .. "\\%"
    })
  end
end

-- Таблиця 3: Перевірка вертикальної розбіжності на 15 контрольних точках (Частина B, Крок 6)
function tab_vertical_discrepancy()
  local rows = read_csv("b_vertical_discrepancy.csv")
  for _, r in ipairs(rows) do
    out({
      r[1],
      esc_tex(r[2]),
      f(r[4], 1),
      f(r[5], 1),
      f(r[6], 2) .. "~пк",
      f(r[8], 1),
      f(r[10], 1),
      "\\textbf{" .. f(r[11], 2) .. "~пк}",
      f(r[12], 1) .. "~пк"
    })
  end
end

-- Таблиця 4: Порівняння алгоритмів StereoBM, StereoSGBM та WLS (Частина C, Крок 7)
function tab_bm_vs_sgbm()
  local rows = read_csv("c_bm_vs_sgbm.csv")
  for _, r in ipairs(rows) do
    out({
      esc_tex(r[1]),
      r[2] .. " / " .. r[3] .. " / " .. r[4],
      f(r[5], 1) .. "\\%",
      f(r[6], 1) .. "~мс",
      f(r[7], 3) .. "~пк",
      f(r[8], 2) .. "~пк",
      esc_tex(r[9])
    })
  end
end

-- Таблиця 5: Вплив параметрів StereoSGBM на якість карти диспаратності (Частина C, Крок 8)
function tab_grid_summary()
  local rows = read_csv("c_grid_summary.csv")
  for _, r in ipairs(rows) do
    local is_opt = (r[1] == "192" and r[2] == "9" and r[3] == "10")
    local nd_s = is_opt and ("\\textbf{" .. r[1] .. "}") or r[1]
    local bs_s = is_opt and ("\\textbf{" .. r[2] .. "}") or r[2]
    local ur_s = is_opt and ("\\textbf{" .. r[3] .. "}") or r[3]
    out({
      nd_s,
      bs_s,
      ur_s,
      f(r[4], 1) .. "\\%",
      f(r[5], 1) .. "~мс",
      f(r[6], 3) .. "~пк",
      f(r[7], 2) .. "~пк",
      esc_tex(r[8])
    })
  end
end

-- Таблиця 6: Верифікація стереоглибини за об'єктами сцени (Частина D, Крок 11)
function tab_depth_verification()
  local rows = read_csv("d_depth_verification.csv")
  for _, r in ipairs(rows) do
    out({
      r[1],
      esc_tex(r[2]),
      "\\texttt{" .. esc_tex(r[3]) .. "}",
      f(r[4], 1) .. " $\\pm$ " .. f(r[5], 2),
      f(r[6], 1) .. "~см",
      "\\textbf{" .. f(r[7], 1) .. "~см}",
      f(r[8], 1) .. "~см",
      f(r[9], 1) .. "\\%"
    })
  end
end

-- Таблиця 7: Систематичні помилки стереозіставлення по зонах (Частина E, Крок 14)
function tab_error_zones()
  local rows = read_csv("e_error_zones.csv")
  for _, r in ipairs(rows) do
    out({
      "\\textbf{" .. esc_tex(r[1]) .. "}",
      esc_tex(r[2]),
      esc_tex(r[3]),
      f(r[5], 1) .. "\\%",
      f(r[6], 1) .. "\\%",
      esc_tex(r[8]),
      esc_tex(r[9])
    })
  end
end
