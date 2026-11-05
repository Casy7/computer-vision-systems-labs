-- Допоміжний модуль Lua для завантаження та форматування числових даних із CSV (lab02/results)
local RES = "../results/"

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
    local r = {}
    for cell in (line .. ","):gmatch("([^,]*),") do
      r[#r + 1] = cell
    end
    rows[#rows + 1] = r
  end
  f:close()
  if #rows > 0 then
    table.remove(rows, 1) -- видалення заголовка
  end
  return rows
end

local function out(cells)
  tex.sprint(table.concat(cells, " & ") .. " \\\\")
end

local function f(x, d)
  if x == "inf" or x == "+inf" then return "$\\infty$" end
  local v = tonumber(x)
  if not v then return tostring(x) end
  return string.format("%." .. (d or 2) .. "f", v)
end

-- Таблиця 1: Характеристики трьох категорій калібрувального набору
function tab_groups()
  local rows = read_csv("ab_group_stats.csv")
  local name_map = {
    frontal = "Фронтальні (різний масштаб)",
    angles = "Під кутом $20^\\circ$--$45^\\circ$",
    corners = "Кути та периферія сенсора"
  }
  for _, r in ipairs(rows) do
    out({
      name_map[r[1]] or r[1],
      r[2],
      f(r[3], 1) .. "\\%",
      f(r[4], 1) .. "\\% -- " .. f(r[5], 1) .. "\\%"
    })
  end
end

-- Таблиця 2: Порівняння етапів калібрування (без субпікселя, субпіксель, фільтрація)
function tab_calibration_comparison()
  local rows = read_csv("c_calibration_comparison.csv")
  for _, r in ipairs(rows) do
    out({
      r[1],
      r[2],
      f(r[3], 4),
      f(r[4], 1),
      f(r[5], 1),
      f(r[6], 4),
      f(r[7], 1),
      f(r[8], 1)
    })
  end
end

-- Таблиця 3: Коефіцієнти дисторсії камери Google Pixel 5
function tab_distortion()
  local rows = read_csv("c_calibration_comparison.csv")
  for _, r in ipairs(rows) do
    out({
      r[1],
      f(r[9], 4),
      f(r[10], 4),
      f(r[11], 5),
      f(r[12], 5),
      f(r[13], 4)
    })
  end
end

-- Таблиця 4: Залежність оцінки параметрів від кількості кадрів N
function tab_n_dependence()
  local rows = read_csv("d_n_frames_dependence.csv")
  for _, r in ipairs(rows) do
    out({
      r[1],
      f(r[2], 3) .. " $\\pm$ " .. f(r[3], 3),
      f(r[4], 1) .. " $\\pm$ " .. f(r[5], 1),
      f(r[6], 1) .. " $\\pm$ " .. f(r[7], 1),
      f(tonumber(r[4]) / tonumber(r[6]), 4)
    })
  end
end

-- Таблиця 5: Профіль відхилення лінії від ідеальної прямої хорди (Крок 14)
function tab_straight_line()
  local rows = read_csv("d_straight_line_profile.csv")
  for _, r in ipairs(rows) do
    out({
      r[1],
      f(r[2], 2),
      f(r[3], 2),
      f(math.abs(tonumber(r[2])) - math.abs(tonumber(r[3])), 2)
    })
  end
end

-- Таблиця 6: Похибки по окремих кадрах (вибіркові кадри та викид)
function tab_frame_errors_sample()
  local rows = read_csv("c_per_frame_errors.csv")
  -- Покажемо перші 2 з кожної групи + викид
  local counts = {}
  for _, r in ipairs(rows) do
    local grp = r[1]
    counts[grp] = (counts[grp] or 0) + 1
    if counts[grp] <= 3 or r[5] == "OUTLIER" then
      local status_tex = r[5] == "OUTLIER" and "\\textbf{\\color{red}Викид (>2 med)}" or "Норма"
      out({
        r[2]:gsub("%.jpg", ""),
        grp,
        f(r[3], 3),
        f(r[4], 3),
        status_tex
      })
    end
  end
end

