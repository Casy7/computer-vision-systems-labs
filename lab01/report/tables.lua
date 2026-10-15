-- Допоміжний модуль Lua для завантаження та форматування числових даних із CSV (lab01/results)
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
    table.remove(rows, 1) -- заголовок
  end
  return rows
end

local function out(cells)
  tex.sprint(table.concat(cells, " & ") .. " \\\\")
end

local function f(x, d)
  if x == "inf" or x == "inf" then return "$\\infty$" end
  local v = tonumber(x)
  if not v then return tostring(x) end
  return string.format("%." .. (d or 2) .. "f", v)
end

-- Таблиця 1: Стиснення (A/photo.jpg та B/day_light.png)
function tab_compression()
  local rows = read_csv("a_compression.csv")
  local by = { photo = {}, day = {} }
  for _, r in ipairs(rows) do
    if by[r[1]] then
      table.insert(by[r[1]], r)
    end
  end
  local n = math.min(#by.photo, #by.day)
  for i = 1, n do
    local a, b = by.photo[i], by.day[i]
    local lab = a[2] == "PNG" and "PNG (без втрат)" or ("JPEG, Q = " .. a[3])
    out({
      lab,
      f(tonumber(a[4]) / 1024, 1),
      f(a[6], 2),
      f(a[5], 2),
      f(tonumber(b[4]) / 1024, 1),
      f(b[6], 2),
      f(b[5], 2)
    })
  end
end

-- Таблиця 2: Сегментація у колірних просторах RGB та HSV
function tab_spaces()
  local rows = read_csv("b_color_spaces.csv")
  for _, r in ipairs(rows) do
    out({
      r[1],
      f(r[2], 3),
      f(r[3], 3),
      f(r[4], 3),
      f(r[5], 4),
      f(r[6], 3)
    })
  end
end

-- Таблиця 3: Оптимальні порогові межі за каналами
function tab_bounds()
  local rows = read_csv("b_bounds.csv")
  for _, r in ipairs(rows) do
    out({
      r[1],
      r[2],
      "[" .. r[3] .. ", " .. r[4] .. "]",
      "[" .. r[5] .. ", " .. r[6] .. "]"
    })
  end
end

-- Таблиця 4: Інтерполяція при масштабуванні x8
function tab_interp()
  local per = read_csv("c_interpolation.csv")
  local mean = read_csv("c_interpolation_mean.csv")
  for _, m in ipairs(mean) do
    local v = {}
    for _, r in ipairs(per) do
      if r[2] == m[1] then
        v[#v + 1] = f(r[3], 2)
      end
    end
    out({
      m[1],
      v[1] or "-",
      v[2] or "-",
      v[3] or "-",
      f(m[2], 2) .. " $\\pm$ " .. f(m[3], 2)
    })
  end
end

-- Таблиця 5: Перекриття пар ракурсів angle1-angle10
function tab_hw_overlap()
  local rows = read_csv("hw_overlap.csv")
  for _, r in ipairs(rows) do
    out({
      r[1],
      r[2],
      r[3],
      f(r[4], 1) .. "\\%"
    })
  end
end

-- Таблиця 6: Характеристики 20 кадрів власного набору
function tab_hw_set()
  local rows = read_csv("hw_set.csv")
  for _, r in ipairs(rows) do
    out({
      r[1],
      "1/" .. r[2],
      r[3],
      f(r[4], 1),
      f(r[5], 1),
      r[6]
    })
  end
end

