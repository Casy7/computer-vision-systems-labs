-- Допоміжний модуль Lua для завантаження та форматування даних CSV у LaTeX-таблиці Лаб 7
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

-- Таблиця 1: Опис 10 власних зображень та перевірка бінарних масок (Частина A, Кроки 1--2)
function tab_dataset_masks()
  local rows = read_csv("a_dataset_masks.csv")
  for _, r in ipairs(rows) do
    local area_s = f(r[8], 2) .. "\\%"
    if tonumber(r[8]) and tonumber(r[8]) < 10.0 then
      area_s = "\\textbf{" .. area_s .. "}"
    end
    out({
      r[1],
      esc_tex(r[2]),
      esc_tex(r[3]),
      "\\texttt{" .. esc_tex(r[4]) .. "}",
      esc_tex(r[5]),
      r[6] .. " $\\to$ " .. r[7],
      area_s
    })
  end
end

-- Таблиця 2: Зведене порівняння 4 методів сегментації (Частина B, Крок 7)
function tab_methods_summary()
  local rows = read_csv("b_methods_summary.csv")
  for _, r in ipairs(rows) do
    local is_gc = r[1]:find("GrabCut") ~= nil
    local mean_s = is_gc and ("\\textbf{" .. f(r[2], 3) .. "}") or f(r[2], 3)
    local dice_s = is_gc and ("\\textbf{" .. f(r[6], 3) .. "}") or f(r[6], 3)
    out({
      esc_tex(r[1]),
      mean_s,
      f(r[3], 3),
      f(r[4], 3),
      f(r[5], 3),
      dice_s,
      f(r[8], 1) .. "~мс"
    })
  end
end

-- Таблиця 3: Детальні значення IoU по кожному з 10 зображень (Частина B)
function tab_per_image_iou()
  local rows = read_csv("b_per_image_iou.csv")
  for _, r in ipairs(rows) do
    local vals = {tonumber(r[5]), tonumber(r[6]), tonumber(r[7]), tonumber(r[8])}
    local best = math.max(vals[1], vals[2], vals[3], vals[4])
    local cells = {r[1], esc_tex(r[2]), f(r[4], 2) .. "\\%"}
    for i = 1, 4 do
      local s = f(vals[i], 3)
      if math.abs(vals[i] - best) < 1e-6 then
        s = "\\textbf{" .. s .. "}"
      end
      cells[#cells + 1] = s
    end
    out(cells)
  end
end

-- Таблиця 4: Демонстрація парадоксу точності за дисбалансу класів (Частина C, Крок 8)
function tab_accuracy_paradox()
  local rows = read_csv("c_accuracy_paradox.csv")
  for _, r in ipairs(rows) do
    out({
      r[1],
      esc_tex(r[2]),
      f(r[3], 2) .. "\\%",
      "\\textbf{" .. f(r[4], 2) .. "\\%}",
      "\\textbf{" .. f(r[5], 3) .. "}",
      f(r[7], 2) .. "\\% / " .. f(r[8], 3),
      f(r[9], 2) .. "\\% / \\textbf{" .. f(r[10], 3) .. "}"
    })
  end
end

-- Таблиця 5: Порівняння конфігурацій детектора HOG+SVM на 30 кадрах INRIA Person (Частина E)
function tab_hog_summary()
  local rows = read_csv("e_hog_summary.csv")
  for i, r in ipairs(rows) do
    local ap_s = (i == 1) and ("\\textbf{" .. f(r[7], 4) .. "}") or f(r[7], 4)
    local f1_s = (i == 1) and ("\\textbf{" .. f(r[6], 3) .. "}") or f(r[6], 3)
    out({
      esc_tex(r[1]),
      r[2],
      r[3],
      r[4],
      f(r[5], 3),
      f1_s,
      ap_s
    })
  end
end

-- Таблиця 6: Ключові порогові точки кривої Precision--Recall після NMS (Частина E)
function tab_pr_thresholds()
  local rows = read_csv("e_pr_thresholds.csv")
  for _, r in ipairs(rows) do
    out({
      f(r[1], 2),
      r[2],
      r[3],
      r[4],
      f(r[5], 3),
      f(r[6], 3),
      f(r[7], 3)
    })
  end
end
