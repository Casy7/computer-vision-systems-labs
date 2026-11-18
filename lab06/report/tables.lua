-- Допоміжний модуль Lua для завантаження та форматування даних CSV у LaTeX-таблиці Лаб 6
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
  return (tostring(s)
    :gsub("&", "\\&")
    :gsub("_", "\\_")
    :gsub("%%", "\\%%")
    :gsub("#", "\\#")
    :gsub("<=", "$\\le$")
    :gsub(">=", "$\\ge$")
    :gsub("<", "$<$")
    :gsub(">", "$>$"))
end

-- Таблиця 1: Параметри відеопослідовностей та стабільність експозиції (Частина A, Кроки 1–2)
function tab_video_stats()
  local rows = read_csv("a_video_stats.csv")
  for _, r in ipairs(rows) do
    out({
      esc_tex(r[1]),
      "\\texttt{" .. esc_tex(r[2]) .. "}",
      r[3],
      f(r[4], 2) .. " / " .. r[5] .. " (" .. f(r[6], 2) .. "~с)",
      f(r[7], 2) .. " $\\pm$ " .. f(r[8], 2),
      f(r[9], 2) .. "--" .. f(r[10], 2),
      f(r[11], 2)
    })
  end
end

-- Таблиця 2: Збереження ознак Lucas-Kanade без та зі зворотною перевіркою (Частина B, Кроки 3–7)
function tab_lk_summary()
  local rows = read_csv("b_lk_summary.csv")
  for _, r in ipairs(rows) do
    local is_fb = r[2]:find("зворотн") ~= nil
    local surv_str = r[4] .. "/" .. r[3] .. " (" .. f(r[5], 1) .. "\\%)"
    local obj_str = r[7] .. "/" .. r[6] .. " (" .. f(r[8], 1) .. "\\%)"
    local bg_str = r[10] .. "/" .. r[9] .. " (" .. f(r[11], 1) .. "\\%)"
    if is_fb then
      surv_str = "\\textbf{" .. surv_str .. "}"
    end
    out({
      esc_tex(r[1]),
      esc_tex(r[2]),
      surv_str,
      obj_str,
      bg_str,
      f(r[12], 2) .. "~мс"
    })
  end
end

-- Таблиця 3: Порівняння розрідженого (PyrLK) та щільного (Farneback) оптичного потоку (Частина C, Крок 10)
function tab_sparse_vs_dense()
  local rows = read_csv("c_sparse_vs_dense.csv")
  for _, r in ipairs(rows) do
    out({
      esc_tex(r[1]),
      r[2],
      esc_tex(r[3]),
      "\\textbf{" .. f(r[4], 2) .. "} $\\pm$ " .. f(r[5], 2) .. "~мс",
      f(r[6], 1),
      esc_tex(r[7])
    })
  end
end

-- Таблиця 4: Етапи віднімання фону MOG2 і пост-обробки на Відео A (Частина D, Кроки 11–13)
function tab_mog2_stages()
  local rows = read_csv("d_mog2_stages.csv")
  for _, r in ipairs(rows) do
    out({
      esc_tex(r[1]),
      r[2] .. "~пк",
      r[3] .. "~пк",
      f(r[4], 1) .. "\\%",
      esc_tex(r[5]),
      esc_tex(r[6])
    })
  end
end

-- Таблиця 5: Точність підрахунку перетинів віртуальної лінії у Відео A (Частина D, Кроки 14–15)
function tab_line_crossing()
  local rows = read_csv("d_line_crossing.csv")
  for _, r in ipairs(rows) do
    local is_main = (r[1] == "620")
    local y_s = is_main and ("\\textbf{" .. r[1] .. "~пк}") or (r[1] .. "~пк")
    out({
      y_s,
      r[2],
      r[3],
      r[4],
      "\\textbf{" .. r[5] .. "}",
      r[6] .. " / " .. r[7],
      "\\textbf{" .. f(r[8], 1) .. "\\%}"
    })
  end
end

-- Таблиця 6: Оцінювання надійності трекерів на розмічених кадрах Відео B (Частина E, Кроки 16–19)
function tab_tracking_metrics()
  local rows = read_csv("e_tracking_metrics.csv")
  for _, r in ipairs(rows) do
    local is_kalman = r[1]:find("Kalman") ~= nil
    local ret_s = r[4] .. "/" .. r[3] .. " (" .. f(r[5], 1) .. "\\%)"
    local sw_s = r[6]
    if is_kalman then
      ret_s = "\\textbf{" .. ret_s .. "}"
      sw_s = "\\textbf{" .. sw_s .. "}"
    end
    out({
      esc_tex(r[1]),
      esc_tex(r[2]),
      ret_s,
      sw_s,
      f(r[7], 1) .. "~пк",
      f(r[8], 1) .. "~пк"
    })
  end
end
